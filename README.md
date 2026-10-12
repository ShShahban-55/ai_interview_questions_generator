 # AI Resume-to-Interview Question Generator

This is my graduation project. You give it a CV (upload a PDF or paste the text) and it writes interview questions for every skill and project it finds in the CV. The questions are grouped by topic, go from easy to hard, and each one comes with an ideal answer. It works in Arabic and English.

The project uses the three techniques I was assigned: Output Parser, RAG and Fine-tuning.

- **Output Parser**: the model has to answer in JSON, so I use Pydantic schemas with LangChain's `PydanticOutputParser` to check the answer. A 3B model often breaks the JSON, so I added some repair steps on top (more on that below).
- **RAG**: for each topic the app looks up reference interview questions in a Chroma vector store (multilingual MiniLM embeddings) and gives them to the model as context.
- **Fine-tuning**: I fine-tuned `Qwen2.5-3B-Instruct` with QLoRA so it gives valid JSON, answers in the language I ask for, and sticks to what is written in the CV.

Author: _your name_
Supervisor: _supervisor name_
University / course: _fill in_

## How it works

My first version sent the CV to the model in one call and asked for 5 questions. That gave only a few questions and ignored most of the CV, so I changed it to 3 steps:

1. **Extraction.** The model reads the CV (first 4000 characters) and returns the candidate level plus a list of topics. Every topic is a skill or a project and has a short quote from the CV as evidence.
2. **Retrieval.** For each topic I get 2 reference questions from Chroma. They are only there for inspiration. If the model copies one of them, that question is dropped.
3. **Generation.** For each topic the model writes k questions (2 to 5, chosen in the sidebar), easy to hard, with ideal answers. Topics are processed in batches and the result is shown grouped by topic.

### Making the parser reliable

The small model makes a lot of small mistakes, for example a misspelled key (`candiate_level`), a level like `Intermediate` that is not in my schema, or JSON that is cut in the middle. So the parsing code does these things:

- repairs broken JSON with `json-repair`
- fixes misspelled keys with `difflib`
- maps values to the allowed ones (`Intermediate` -> `Mid`, and Arabic words like `صعب` -> `Hard`)
- drops questions that are cut off or duplicated
- if one topic fails, the other topics still work

### Training data

I could not find a public dataset of CV -> interview questions, so I built one. The base model runs the same pipeline as the app on real CVs (Kaggle "Resume Dataset", IT and Engineering categories), once in English and once in Arabic, and I keep only the good outputs:

- the evidence for a topic must really be in the CV (at least 80% of its words)
- the number of questions must be exactly k
- the language must match (Arabic ratio check, and no Chinese characters, which sometimes showed up)
- no question copied from the RAG references

The training scripts read the prompts and schemas directly from `app.py`, so training and the app always use the same prompts.

One thing I want to be clear about: this fine-tuning does not teach the model new knowledge. It only teaches it the output format, the language and to stay close to the CV. The knowledge itself comes from the base model and from RAG.

## Files

```
app.py                        the Streamlit app (pipeline, parsers, UI)
build_vectorstore.py          builds the Chroma DB from knowledge/knowledge.json
knowledge/knowledge.json      19 reference entries I wrote for RAG
training/generate_dataset.py  makes train_samples.jsonl from the CVs
training/train_qlora.py       QLoRA training + comparison base vs fine-tuned
tests/                        tests that run without a GPU (they use a fake LLM)
requirements.txt
```

## How to run

You need a GPU. I used the free T4 on Kaggle.

```bash
pip install -r requirements.txt
python build_vectorstore.py
python training/generate_dataset.py --n-cvs 40
python training/train_qlora.py --epochs 2
streamlit run app.py
```

Generating the data took around 1.5 hours and the training around 30-40 minutes on the T4. After training, the app finds the `qlora_adapter/` folder and the "Use fine-tuned model" switch in the sidebar becomes available.

On Kaggle I opened the app with a Cloudflare tunnel: `./cloudflared tunnel --url http://localhost:8501`. If peft gives a `torchao` version error, run `pip uninstall -y torchao`.

To run the tests (no GPU needed):

```bash
python tests/test_pipeline.py
python tests/test_training_data.py
```

## Fine-tuning settings

- 4-bit NF4 quantization with double quantization, fp16 compute
- LoRA r=16, alpha=32, dropout 0.05 on the attention and MLP layers (about 29.9M trainable parameters, 0.96% of the model)
- learning rate 2e-4, effective batch size 8, 2 epochs, max length 2400
- the loss is calculated only on the answer tokens, not on the prompt
- validation uses whole CVs that are not in the training set

## Current status

Done: the app pipeline, the output parser, RAG, the Arabic/English UI, the tests, and a small test run of the training (35 samples) that saved a working adapter.

Still to do: generate the full dataset (40 CVs), run the final training, and fill in this table with the real numbers.

| | Base model | + RAG | + RAG + fine-tuning |
|---|---|---|---|
| Valid JSON on first try (no repair) | | | |
| Relevance to the CV (my rating, 1-5) | | | |
| Topics covered per CV | | | |

## Limitations

- The training data is generated by the base model itself and filtered with simple rules, so it keeps some of the base model's weaknesses.
- The knowledge base is small (19 entries).
- The 3B model sometimes writes generic questions, or mixes English words into Arabic text.
- The app needs a GPU. The 3B model does not fit on Streamlit Community Cloud (about 2.7 GB RAM and no GPU), so I run the demo on Kaggle. The adapter is too big for GitHub, so it will be on the Hugging Face Hub instead.

## Screenshots and demo

_I will add screenshots and the demo video link here._
