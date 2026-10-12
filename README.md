# AI Resume-to-Interview Question Generator

Upload a CV (or paste its text) and get **interview questions for every skill and project in it**, grouped per topic, ordered Easy → Hard, each with an ideal answer. Works in **Arabic and English**.

The project combines three techniques:

| Technique | Role in the project |
|---|---|
| **Output Parser** | Forces the LLM output into typed Pydantic schemas (LangChain `PydanticOutputParser`), with repair layers for the messy JSON a 3B model produces |
| **RAG** | Retrieves reference interview material from a Chroma vector store (multilingual MiniLM embeddings) for each topic |
| **Fine-tuning (QLoRA)** | Adapts `Qwen2.5-3B-Instruct` so it produces schema-valid JSON, in the requested language, grounded in the CV |

> Author: _your name_ · Supervisor: _supervisor name_ · University / Course: _fill in_

## How it works

```
CV text ──► 1. Extraction ──► candidate level + topics (skills / projects, each with CV evidence)
                │
                ▼
        2. RAG retrieval (k=2 reference Q&A per topic, Chroma)
                │
                ▼
        3. Question generation (per topic, batched) ──► Output Parser ──► grouped results
```

1. **Extraction** reads the whole CV (first 4000 chars) and returns the candidate level and a list of topics, each with a short evidence quote from the CV.
2. **Retrieval** fetches reference questions for each topic. They are inspiration only: questions copied from them are dropped (`drop_copies`).
3. **Generation** asks for *k* questions per topic (2–5), Easy → Hard, with ideal answers, in the chosen language.

### Output parser robustness
Pydantic v2 schemas + `PydanticOutputParser`, plus: `json-repair` for truncated/broken JSON, fuzzy key fixing (`candiate_level` → `candidate_level`), enum normalisation (`Intermediate` → `Mid`, Arabic synonyms), dropping truncated or duplicate questions. One failed topic never kills the others.

### Fine-tuning data
There is no public CV→questions dataset, so the training set is built by **self-distillation with rejection sampling**: the base model runs the exact app pipeline on real CVs (Kaggle *Resume Dataset*, IT + Engineering categories), in English and Arabic, and only clean outputs are kept:
- the topic evidence must really appear in the CV (≥80 % of its tokens)
- exactly *k* questions
- language matches (Arabic ratio check; no Chinese/Japanese characters)
- no questions copied from RAG references

Prompts and schemas are read from `app.py`, so training and inference use identical prompts.

**Honest framing:** fine-tuning here teaches *format, discipline and language*, not new knowledge. The knowledge comes from the base model and RAG.

## Repository layout

```
app.py                      Streamlit app (pipeline, parsers, bilingual UI)
build_vectorstore.py        Builds the Chroma DB from knowledge/knowledge.json
knowledge/knowledge.json    19 hand-written reference entries (RAG knowledge base)
training/generate_dataset.py  Builds train_samples.jsonl from CVs
training/train_qlora.py       QLoRA training + base-vs-fine-tuned evaluation
tests/                      GPU-free logic tests (fake LLM)
requirements.txt
```

## Run it

Needs a GPU (developed on a free Kaggle T4). Steps:

```bash
pip install -r requirements.txt
python build_vectorstore.py                      # build the RAG index
python training/generate_dataset.py --n-cvs 40   # ~1.5 h on a T4
python training/train_qlora.py --epochs 2        # ~30-40 min, writes qlora_adapter/
streamlit run app.py                             # sidebar: toggle RAG / fine-tuned model
```

On Kaggle, expose the app with a Cloudflare quick tunnel:
`./cloudflared tunnel --url http://localhost:8501`.
If peft complains about `torchao`: `pip uninstall -y torchao`.

Tests (no GPU needed):

```bash
python tests/test_pipeline.py
python tests/test_training_data.py
```

## Fine-tuning setup

4-bit NF4 QLoRA (double quantisation, fp16 compute), LoRA r=16 / alpha=32 / dropout 0.05 on all attention + MLP projections (~29.9 M trainable params, 0.96 %), LR 2e-4, effective batch 8, 2 epochs, max length 2400, loss only on the answer tokens. Validation uses whole CVs that never appear in training.

## Status and results

- ✅ Full app pipeline, output parser, RAG, bilingual UI, tests
- ✅ Training pipeline verified end-to-end on a small run (35 samples) — adapter saved and loadable
- ⏳ Full-scale dataset (40 CVs) and final training run
- ⏳ Evaluation (fill in after the final run):

| Metric | Base | + RAG | + RAG + fine-tuning |
|---|---|---|---|
| Valid JSON on first try (no repair) | _ | _ | _ |
| Questions rated relevant to the CV (human, 1–5) | _ | _ | _ |
| Topics covered per CV | _ | _ | _ |

## Limitations

- Training data is self-generated and filtered by rules, so it inherits the base model's weaknesses.
- The knowledge base has only 19 entries.
- A 3B model sometimes mixes English technical terms into Arabic text (intended for terms like "Docker") or produces generic questions.
- The app needs a GPU; the 3B model does not fit Streamlit Community Cloud (~2.7 GB RAM, no GPU). The LoRA adapter is too large for GitHub and should be hosted on the Hugging Face Hub.

## Screenshots
_Add screenshots / demo video link in `docs/`._
