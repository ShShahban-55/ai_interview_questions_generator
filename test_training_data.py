"""Tests for dataset filtering and training-data preparation. No GPU / model downloads needed.

Run from anywhere:  python tests/test_training_data.py
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "training" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = load("generate_dataset")
trn = load("train_qlora")

CV = ("Ahmed Backend Developer. Skills: Python, Django, Docker. "
      "Project: Payment API with JWT authentication and rate limiting, deployed with Docker.")


def make_fake(lang):
    def fake(pairs, max_new, on_batch=None):
        outs = []
        for system, user in pairs:
            if system == gen.SYSTEM_EXTRACT:
                outs.append(json.dumps({"candidate_level": "Mid", "topics": [
                    {"name": "Django", "type": "skill", "evidence": "Python, Django, Docker"},
                    {"name": "Payment API", "type": "project", "evidence": "Payment API with JWT authentication"},
                    {"name": "Docker", "type": "skill", "evidence": "deployed with Docker"},
                    {"name": "Kubernetes", "type": "skill",
                     "evidence": "managed 40 kubernetes clusters in production"}]}))   # not in the CV
            elif "Topic: Django" in user:
                if lang == "en":
                    outs.append('{"questions":[{"question":"How did you use the ORM?","difficulty":"Easy","ideal_answer":"Querysets."},'
                                '{"question":"How to avoid N+1?","difficulty":"Medium","ideal_answer":"select_related."},'
                                '{"question":"How did you scale it?","difficulty":"Hard","ideal_answer":"Caching."}]}')
                else:
                    outs.append('{"questions":[{"question":"كيف استخدمت Django ORM في مشروعك؟","difficulty":"Easy","ideal_answer":"باستخدام النماذج والاستعلامات."},'
                                '{"question":"كيف تتجنب مشكلة N+1 في Django؟","difficulty":"Medium","ideal_answer":"استخدم select_related لتقليل الاستعلامات."},'
                                '{"question":"كيف وسّعت نطاق التطبيق؟","difficulty":"Hard","ideal_answer":"باستخدام التخزين المؤقت ونسخ القراءة."}]}')
            elif "Topic: Payment API" in user:     # contains a CJK character -> must be rejected
                outs.append('{"questions":[{"question":"JWT design?","difficulty":"Easy","ideal_answer":"ت冻结 tokens"},'
                            '{"question":"Rate limiting?","difficulty":"Hard","ideal_answer":"Counters."},'
                            '{"question":"x?","difficulty":"Medium","ideal_answer":"y"}]}')
            else:                                  # Docker: 1 question instead of 3 -> must be rejected
                outs.append('{"questions":[{"question":"Why Docker?","difficulty":"Easy","ideal_answer":"Builds."}]}')
        if on_batch:
            on_batch(len(pairs), len(pairs))
        return outs
    return fake


def run(lang):
    log = []

    def llm(pairs, m, ob=None):
        out = make_fake(lang)(pairs, m, ob)
        log.append(pairs)
        return out

    return gen.collect_samples(CV, "Backend Developer", lang, 3, 8, llm,
                               lambda q: ["Question: Q? Ideal answer: A"], log)


def test_english_run_keeps_only_clean_samples():
    samples, rej = run("en")
    assert [s["task"] for s in samples] == ["extract", "topic"]
    topics = [t["name"] for t in json.loads(samples[0]["target"])["topics"]]
    assert "Kubernetes" not in topics and topics == ["Django", "Payment API", "Docker"]   # grounded only
    assert rej["ungrounded_topic"] == 1 and rej["language"] == 1 and rej["wrong_question_count"] == 1


def test_arabic_run_skips_extraction_and_checks_language():
    samples, rej = run("ar")
    assert [s["task"] for s in samples] == ["topic"]          # extraction is only taken from the English run
    assert len(json.loads(samples[0]["target"])["questions"]) == 3


def test_training_prompt_is_the_app_prompt():
    samples, _ = run("en")
    assert samples[0]["system"] == gen.SYSTEM_EXTRACT
    assert samples[1]["system"] == gen.SYSTEM_Q and "Number of questions: 3" in samples[1]["user"]


class StubTok:                                # tiny fake tokenizer: every word is a token
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [hash(w) % 1000 for w in text.split()]}


def test_prompt_tokens_are_masked_and_answer_is_trained():
    sample = {"task": "topic", "system": "sys prompt", "user": "user text here", "target": '{"questions": []}'}
    e = trn.encode_sample(StubTok(), sample, 2400)
    n_prompt = sum(1 for x in e["labels"] if x == -100)
    assert 0 < n_prompt < len(e["labels"])
    assert e["labels"][n_prompt:] == e["input_ids"][n_prompt:]
    assert trn.encode_sample(StubTok(), sample, 3) is None        # too long -> dropped


def test_strict_validity_metric():
    ok = '{"questions": [{"question": "q", "difficulty": "Easy", "ideal_answer": "a"}]}'
    assert trn.strict_ok(ok, "topic")
    assert not trn.strict_ok("```json\n" + ok + "\n```", "topic")          # fenced output is not "valid first try"
    assert not trn.strict_ok('{"questons": []}', "topic")                   # misspelled key
    assert not trn.strict_ok(ok.replace("Easy", "Moderate"), "topic")       # value outside the schema


def test_validation_split_never_shares_a_cv():
    rows = [{"idx": i, "task": "topic"} for i in range(10) for _ in range(3)]
    train, val = trn.split_by_cv(rows)
    assert {r["idx"] for r in train}.isdisjoint({r["idx"] for r in val}) and val


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
