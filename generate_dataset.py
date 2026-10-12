"""Generate the fine-tuning dataset (chat-style samples) from real CVs.

The samples use EXACTLY the same prompts, schemas and parsers as the app (they are read from app.py),
so the model is trained on the same format it will see at inference time.

Pipeline per CV and per language (en / ar):
    1. run the app pipeline with the base model (extract topics -> RAG per topic -> questions)
    2. keep only clean samples:
         - topic evidence must really come from the CV (grounding check)
         - the number of questions must match the request
         - the language must match (no mixed scripts, no CJK characters)
         - questions copied from the RAG references were already dropped by the pipeline

Usage (from the repo root, GPU required):
    python training/generate_dataset.py --n-cvs 40 --resume-csv /path/to/Resume.csv

The script is resumable: finished (cv, language) runs are recorded in runs_done.txt.
"""
import argparse
import difflib
import glob
import html
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Literal

from json_repair import repair_json
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = ROOT / "train_samples.jsonl"
DONE_FILE = ROOT / "runs_done.txt"

# Prompts, schemas, parsers and run_pipeline come from app.py (single source of truth).
_src = (ROOT / "app.py").read_text(encoding="utf-8")
exec(_src.split("# ===== HELPERS START =====")[1].split("# ===== HELPERS END =====")[0], globals())

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def arabic_ratio(text):
    letters = [c for c in text if c.isalpha()]
    return sum("؀" <= c <= "ۿ" for c in letters) / max(len(letters), 1)


def lang_ok(q, lang):
    if CJK.search(q.question + q.ideal_answer):
        return False
    if lang == "ar":
        return arabic_ratio(q.question) >= 0.3 and arabic_ratio(q.ideal_answer) >= 0.25
    return arabic_ratio(q.question + q.ideal_answer) <= 0.05


def grounded(evidence, cv_words):
    toks = re.findall(r"\w+", evidence.lower())
    return bool(toks) and sum(t in cv_words for t in toks) / len(toks) >= 0.8


def guess_job(raw_resume, category):
    head = re.split(r"\b(?:Summary|Professional|Career|Objective|Highlights|Skills|Profile|Experience|"
                    r"Qualifications|Overview|Executive)\b", raw_resume[:200], maxsplit=1, flags=re.I)[0]
    head = " ".join(head.split())
    if 1 <= len(head.split()) <= 6:
        return head.title()
    return category.replace("-", " ").title() + " Role"


def collect_samples(cv, job, lang, k, max_topics, llm, retrieve, log):
    """Run the app pipeline and keep only the clean samples. `log` records every llm call."""
    ext, sections, _ = run_pipeline(cv, job, lang, k, max_topics, True, llm, retrieve)
    cv_words = set(re.findall(r"\w+", cv.lower()))
    good = [tp for tp in ext.topics if grounded(tp.evidence, cv_words)]
    samples, rej = [], Counter()
    if lang == "en":        # topic extraction does not depend on the language: take it once
        if len(good) >= 3:
            sys_, usr_ = log[0][0]
            samples.append({"task": "extract", "system": sys_, "user": usr_,
                            "target": json.dumps({"candidate_level": ext.candidate_level,
                                                  "topics": [tp.model_dump() for tp in good]},
                                                 ensure_ascii=False)})
        else:
            rej["few_grounded_topics"] += 1
    for sec, (sys_, usr_) in zip(sections, log[1]):
        if sec["topic"] not in good:
            rej["ungrounded_topic"] += 1
            continue
        if not sec["ok"]:
            rej["topic_failed"] += 1
            continue
        qs = sec["questions"]
        if len(qs) != k:
            rej["wrong_question_count"] += 1
            continue
        if not all(lang_ok(q, lang) for q in qs):
            rej["language"] += 1
            continue
        samples.append({"task": "topic", "system": sys_, "user": usr_,
                        "target": json.dumps({"questions": [q.model_dump() for q in qs]},
                                             ensure_ascii=False)})
    return samples, rej


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-cvs", type=int, default=40, help="number of CVs to process (each in en + ar)")
    ap.add_argument("--resume-csv", default=os.environ.get("RESUME_CSV"),
                    help="path to Resume.csv (Kaggle 'Resume Dataset'); auto-detected on Kaggle")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cv-chars", type=int, default=4000)
    args = ap.parse_args()

    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    csv_path = args.resume_csv or next(iter(glob.glob("/kaggle/input/**/Resume.csv", recursive=True)), None)
    if not csv_path:
        raise SystemExit("Resume.csv not found. Pass --resume-csv or add the Kaggle 'Resume Dataset'.")
    df = pd.read_csv(csv_path)
    tech = df[df["Category"].isin(["INFORMATION-TECHNOLOGY", "ENGINEERING"])].copy()
    tech["job"] = [guess_job(r, c) for r, c in zip(tech["Resume_str"], tech["Category"])]
    tech["cv"] = tech["Resume_str"].str.replace(r"\s+", " ", regex=True).str.strip().str[:args.cv_chars]
    tech = tech.sample(frac=1, random_state=42).reset_index(drop=True)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                model_kwargs={"device": "cpu"})
    retriever = Chroma(persist_directory=str(ROOT / "chroma_db"),
                       embedding_function=emb).as_retriever(search_kwargs={"k": 2})
    retrieve = lambda q: [d.page_content for d in retriever.invoke(q)]

    tok = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16,
                                                 device_map="auto").eval()

    def make_llm(log):
        def llm(pairs, max_new_tokens, on_batch=None):
            outs = []
            for b in range(0, len(pairs), args.batch):
                chunk = pairs[b:b + args.batch]
                prompts = [tok.apply_chat_template(
                    [{"role": "system", "content": s}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True) for s, u in chunk]
                inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                         temperature=0.5, top_p=0.9, pad_token_id=tok.pad_token_id)
                outs += tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                if on_batch:
                    on_batch(min(b + args.batch, len(pairs)), len(pairs))
            log.append(pairs)
            return outs
        return llm

    done = set(DONE_FILE.read_text().split()) if DONE_FILE.exists() else set()
    kept, rejects, t0 = Counter(), Counter(), time.time()

    for idx in tqdm(range(args.n_cvs)):
        rng = random.Random(idx)                 # same k / max_topics for both languages
        k, max_topics = rng.choice([2, 3, 4]), rng.choice([5, 8, 10])
        row = tech.iloc[idx]
        for lang in ("en", "ar"):
            key = f"{idx}:{lang}"
            if key in done:
                continue
            log = []
            try:
                samples, rej = collect_samples(row.cv, row.job, lang, k, max_topics,
                                               make_llm(log), retrieve, log)
            except PipelineError:
                rejects["extraction_failed"] += 1
                samples, rej = [], Counter()
            rejects.update(rej)
            with open(OUT_FILE, "a", encoding="utf-8") as f:
                for s in samples:
                    kept[f'{s["task"]}:{lang}'] += 1
                    f.write(json.dumps({**s, "idx": idx, "lang": lang}, ensure_ascii=False) + "\n")
            with open(DONE_FILE, "a") as f:
                f.write(key + "\n")

    total = sum(1 for _ in open(OUT_FILE, encoding="utf-8")) if OUT_FILE.exists() else 0
    print(f"\nThis run: kept={dict(kept)} | rejected={dict(rejects)} | {(time.time() - t0) / 60:.1f} min")
    print(f"Total samples in {OUT_FILE.name}: {total}")


if __name__ == "__main__":
    main()
