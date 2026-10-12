"""QLoRA fine-tuning of Qwen2.5-3B-Instruct on the generated samples (single GPU, plain PyTorch loop).

Only the assistant answer contributes to the loss (the prompt tokens are masked with -100).
Validation uses whole CVs that never appear in training. At the end the script compares the base model
with the fine-tuned one on held-out prompts: how often is the raw output already valid JSON that matches
the schema, with no repair at all?

Usage (from the repo root, GPU required):
    python training/train_qlora.py --epochs 2

Output: qlora_adapter/  (loaded automatically by app.py)
"""
import argparse
import difflib
import html
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Literal

from json_repair import repair_json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent

# Same prompts / schemas as the app (single source of truth).
_src = (ROOT / "app.py").read_text(encoding="utf-8")
exec(_src.split("# ===== HELPERS START =====")[1].split("# ===== HELPERS END =====")[0], globals())


def encode_sample(tok, s, max_len):
    """Prompt tokens (masked) + answer tokens (trained). Returns None if the sample is too long."""
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["user"]}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(s["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(t_ids) > max_len:
        return None
    return {"input_ids": p_ids + t_ids, "labels": [-100] * len(p_ids) + t_ids, "task": s["task"]}


def strict_ok(raw, task):
    """True if the raw model output is itself valid JSON matching the schema (no repair)."""
    try:
        obj = json.loads(raw.strip())
        (Extraction if task == "extract" else TopicQuestions).model_validate(obj)
        return True
    except Exception:
        return False


def lr_lambda_factory(total, warm):
    return lambda s: (s + 1) / warm if s < warm else max(0.05, (total - s) / max(total - warm, 1))


def split_by_cv(rows, seed=0):
    """Hold out whole CVs for validation so no CV leaks between train and validation."""
    idxs = sorted({r["idx"] for r in rows})
    random.Random(seed).shuffle(idxs)
    val_idx = set(idxs[:max(1, len(idxs) // 10)])
    return [r for r in rows if r["idx"] not in val_idx], [r for r in rows if r["idx"] in val_idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--samples", default=str(ROOT / "train_samples.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "qlora_adapter"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--max-len", type=int, default=2400)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(l) for l in open(args.samples, encoding="utf-8")]
    train_rows, val_rows = split_by_cv(rows)
    print(f"samples: train={len(train_rows)} | val={len(val_rows)}")
    if len(train_rows) < 60:
        print("WARNING: fewer than 60 training samples. Generate more data first for a meaningful run.")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    train = [e for e in (encode_sample(tok, s, args.max_len) for s in train_rows) if e]
    val = [e for e in (encode_sample(tok, s, args.max_len) for s in val_rows) if e]
    print(f"after dropping long samples: train={len(train)} | val={len(val)}")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb,
                                                 device_map={"": 0}, torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    model.config.use_cache = False

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = math.ceil(len(train) / args.accum) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(total_steps, max(1, int(0.06 * total_steps))))
    scaler = torch.cuda.amp.GradScaler()
    dev = "cuda:0"

    def batch_loss(ex):
        ids = torch.tensor([ex["input_ids"]], device=dev)
        lab = torch.tensor([ex["labels"]], device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            return model(input_ids=ids, labels=lab).loss

    model.train()
    t0, step, run_loss, since = time.time(), 0, 0.0, 0
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train)
        for i, ex in enumerate(train):
            loss = batch_loss(ex) / args.accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
            if (i + 1) % args.accum == 0 or i == len(train) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                since += 1
                if step % 5 == 0 or step == total_steps:
                    print(f"epoch {epoch + 1}/{args.epochs} | step {step}/{total_steps} | "
                          f"loss {run_loss / since:.4f} | {(time.time() - t0) / 60:.1f} min")
                    run_loss, since = 0.0, 0

    model.eval()
    with torch.no_grad():
        vl = [batch_loss(e).item() for e in val]
    print(f"\nVal loss: {sum(vl) / len(vl):.4f}")

    model.save_pretrained(args.out_dir)
    size_mb = sum(os.path.getsize(os.path.join(args.out_dir, f)) for f in os.listdir(args.out_dir)) / 1e6
    print(f"Adapter saved to {args.out_dir}/ ({size_mb:.0f} MB)")

    # ---- base vs fine-tuned: is the RAW output valid JSON matching the schema? ----
    tok.padding_side = "left"
    model.config.use_cache = True
    probe = ([s for s in val_rows if s["task"] == "topic"][:12]
             + [s for s in val_rows if s["task"] == "extract"][:4])

    def generate(rows_, adapter_on):
        outs = []
        for b in range(0, len(rows_), 4):
            chunk = rows_[b:b + 4]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": s["system"]}, {"role": "user", "content": s["user"]}],
                tokenize=False, add_generation_prompt=True) for s in chunk]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(dev)
            ctx = nullcontext() if adapter_on else model.disable_adapter()
            with ctx, torch.no_grad():
                o = model.generate(**inputs, max_new_tokens=700, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    for name, on in (("base model", False), ("fine-tuned model", True)):
        outs = generate(probe, on)
        ok = sum(strict_ok(o, s["task"]) for o, s in zip(outs, probe))
        print(f"{name}: valid JSON on the first try = {ok}/{len(probe)} ({100 * ok / max(len(probe), 1):.0f}%)")


if __name__ == "__main__":
    main()
