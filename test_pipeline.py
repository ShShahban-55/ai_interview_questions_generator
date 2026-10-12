"""Logic tests that need no GPU and no model: a fake LLM returns deliberately messy outputs.

Run from anywhere:  python tests/test_pipeline.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_src = (ROOT / "app.py").read_text(encoding="utf-8")
_block = _src.split("# ===== HELPERS START =====")[1].split("# ===== HELPERS END =====")[0]
_pre = ("import json, html, re, difflib\nfrom typing import List, Literal\nfrom pydantic import BaseModel\n"
        "from json_repair import repair_json\nfrom langchain_core.output_parsers import PydanticOutputParser\n")
ns = {}
exec(_pre + _block, ns)

run_pipeline, to_text, PipelineError = ns["run_pipeline"], ns["to_text"], ns["PipelineError"]
parse_extraction, parse_topic_questions = ns["parse_extraction"], ns["parse_topic_questions"]
SYSTEM_EXTRACT = ns["SYSTEM_EXTRACT"]

CV = "Skills: Python, Django, Docker. Project: payment API with JWT."
REF = ["Question: What is RAG and why use it instead of only fine-tuning? Ideal answer: blah"]


def fake_llm(pairs, max_new_tokens, on_batch=None):
    outs = []
    for i, (system, user) in enumerate(pairs):
        if system == SYSTEM_EXTRACT:
            # misspelled key, level synonym, duplicated topic, Arabic type, bare string topic
            outs.append('{"candiate_level": "Intermediate", "topics": ['
                        '{"name": "Django", "type": "skill", "evidence": "Django"},'
                        '{"name": "django", "type": "skill"},'
                        '{"name": "Payment API", "type": "مشروع", "evidence": "payment API with JWT"},'
                        '{"name": "Docker", "type": "tool"}, "Redis"]}')
        elif "Topic: Django" in user:       # a bare list instead of an object
            outs.append('[{"question": "How did you use the ORM?", "difficulty": "Easy", "ideal_answer": "a1"},'
                        '{"question": "How to avoid N+1 queries?", "difficulty": "Moderate", "ideal_answer": "a2"},'
                        '{"question": "How did you scale it?", "difficulty": "Hard", "ideal_answer": "a3"}]')
        elif "Topic: Payment API" in user:  # truncated JSON (last question cut)
            outs.append('{"questions": [{"question": "JWT design?", "difficulty": "متوسط", "ideal_answer": "j1"},'
                        '{"question": "Rate limiting?", "difficulty": "صعب", "ideal_answer": "j2"},'
                        '{"question": "cut", "diffic')
        elif "Topic: Docker" in user:       # no JSON at all
            outs.append("sorry, I cannot do that")
        else:                               # Redis: only copies a RAG reference question
            outs.append('{"questions": [{"question": "What is RAG and why use it instead of only fine-tuning?", '
                        '"difficulty": "Easy", "ideal_answer": "x"}]}')
        if on_batch:
            on_batch(i + 1, len(pairs))
    return outs


def test_full_pipeline_with_messy_outputs():
    stages = []
    ext, sections, stats = run_pipeline(CV, "Backend", "ar", 3, 8, True, fake_llm, lambda q: REF,
                                        lambda *a: stages.append(a))
    assert ext.candidate_level == "Mid"                                   # "Intermediate" normalised
    assert [(t.name, t.type) for t in ext.topics] == [
        ("Django", "skill"), ("Payment API", "project"), ("Docker", "skill"), ("Redis", "skill")]
    by_name = {s["topic"].name: s for s in sections}
    assert by_name["Django"]["ok"] and [q.difficulty for q in by_name["Django"]["questions"]] == ["Easy", "Medium", "Hard"]
    assert by_name["Payment API"]["ok"] and len(by_name["Payment API"]["questions"]) == 2   # truncated one dropped
    assert not by_name["Docker"]["ok"]                                    # one failure must not kill the rest
    assert not by_name["Redis"]["ok"] and "copied" in by_name["Redis"]["error"]
    assert stats["calls"] == 5 and stats["failed"] == 2
    assert stages[0] == ("extract", 0, 1) and stages[-1][0] == "generate"


def test_export_text_is_grouped_by_topic():
    ext, sections, _ = run_pipeline(CV, "Backend", "ar", 3, 8, True, fake_llm, lambda q: REF)
    text = to_text(ext, sections, "ar", "Backend")
    assert text.index("Django") < text.index("Payment API")
    assert "س1)" in text and "س5)" in text


def test_extraction_failure_keeps_the_raw_output():
    try:
        run_pipeline(CV, "x", "en", 3, 8, False, lambda p, m, o=None: ["no json here"], lambda q: [])
    except PipelineError as e:
        assert e.raw == "no json here"
    else:
        raise AssertionError("PipelineError expected")


def test_rag_can_be_disabled():
    seen = []
    run_pipeline(CV, "Backend", "en", 3, 8, False, fake_llm, lambda q: seen.append(q) or REF)
    assert seen == []


def test_parsers_handle_arabic_values_and_clean_json():
    r, info = parse_topic_questions(
        '{"questions": [{"question": "كيف؟", "difficulty": "صعب", "ideal_answer": "هكذا"}]}', 3)
    assert r.questions[0].difficulty == "Hard" and info["normalized"] and not info["repaired"]
    r, info = parse_topic_questions(
        '{"questions": [{"question": "q", "difficulty": "Easy", "ideal_answer": "a"}]}', 3)
    assert not info["repaired"] and not info["normalized"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)} tests passed")
