import json, os, time, html, re, difflib
from contextlib import nullcontext
from typing import List, Literal

import torch
import streamlit as st
from pydantic import BaseModel
from json_repair import repair_json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

st.set_page_config(page_title="Interview Question Generator", page_icon="🎯", layout="wide")

BASE = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "qlora_adapter"          # بيتعمل في خطوة الـ Fine-tuning
BATCH = 4
CV_CHARS = 4000

# ===== HELPERS START =====
# ---------- Prompts (نفس الصيغة اللي هنتدرّب عليها) ----------
SYSTEM_EXTRACT = (
    "You are an expert technical recruiter. Read the candidate CV and return ONLY valid JSON, no extra text. "
    "Keys: candidate_level (exactly one of: Junior, Mid, Senior) and topics (a list, at most the requested "
    "maximum). Each topic has: name (a specific skill, tool or project taken from the CV), "
    "type (exactly one of: skill, project) and evidence (a short phrase copied from the CV). "
    "Cover every important skill and project in the CV.")

SYSTEM_Q = (
    "You are an expert technical interviewer. Given a candidate CV, one topic taken from that CV and optional "
    "reference material, return ONLY valid JSON, no extra text. Key: questions, a list whose length equals the "
    "requested number, ordered from easy to hard. Each item has: question (about the candidate's own experience "
    "with this topic; never copy reference questions; ignore references unrelated to the topic), "
    "difficulty (exactly one of: Easy, Medium, Hard) and ideal_answer (concise and technically correct).")


def lang_line(lang):
    if lang == "ar":
        return ("Output language: Arabic. Write question and ideal_answer in Arabic "
                "(keep technical terms such as Django or SQL in English). "
                "Keep difficulty values in English exactly as specified.")
    return "Output language: English."


def build_extract_user(cv, job, max_topics):
    return f"CV:\n{cv}\n\nJob: {job}\n\nMaximum topics: {max_topics}"


def build_topic_user(cv, job, topic, ctx, k, lang):
    ref = "\n".join(f"- {c}" for c in ctx) or "- (none)"
    ev = f"\nEvidence: {topic.evidence}" if topic.evidence else ""
    return (f"CV:\n{cv}\n\nJob: {job}\n\nTopic: {topic.name} ({topic.type}){ev}\n\n"
            f"Reference material:\n{ref}\n\nNumber of questions: {k}\n\n{lang_line(lang)}")


# ---------- Schemas ----------
class Topic(BaseModel):
    name: str
    type: Literal["skill", "project"]
    evidence: str = ""


class Extraction(BaseModel):
    candidate_level: Literal["Junior", "Mid", "Senior"]
    topics: List[Topic]


class TQuestion(BaseModel):
    question: str
    difficulty: Literal["Easy", "Medium", "Hard"]
    ideal_answer: str


class TopicQuestions(BaseModel):
    questions: List[TQuestion]


extraction_parser = PydanticOutputParser(pydantic_object=Extraction)
questions_parser = PydanticOutputParser(pydantic_object=TopicQuestions)

LEVELS = {"junior": "Junior", "entry": "Junior", "beginner": "Junior", "مبتدئ": "Junior", "جونيور": "Junior",
          "mid": "Mid", "mid-level": "Mid", "middle": "Mid", "intermediate": "Mid", "متوسط": "Mid",
          "senior": "Senior", "expert": "Senior", "advanced": "Senior", "متقدم": "Senior",
          "خبير": "Senior", "سينيور": "Senior"}
DIFFS = {"easy": "Easy", "basic": "Easy", "سهل": "Easy",
         "medium": "Medium", "moderate": "Medium", "intermediate": "Medium", "متوسط": "Medium",
         "hard": "Hard", "difficult": "Hard", "advanced": "Hard", "صعب": "Hard"}
TYPES = {"skill": "skill", "skills": "skill", "tool": "skill", "technology": "skill", "مهارة": "skill",
         "project": "project", "projects": "project", "مشروع": "project"}


# ---------- Parsing helpers ----------
def _fix_keys(d, expected):
    """يصلّح أخطاء الإملاء في أسماء الحقول (candiate_level -> candidate_level)."""
    out = {}
    for k, v in d.items():
        m = k if k in expected else (difflib.get_close_matches(k, expected, n=1, cutoff=0.8) or [k])[0]
        out.setdefault(m, v)
    return out


def extract_obj(raw, list_key=None):
    """نص الموديل -> (dict, repaired). بيتعامل مع JSON مكسور أو list مباشرة."""
    i_obj, i_arr = raw.find("{"), raw.find("[")
    if list_key and i_arr != -1 and (i_obj == -1 or i_arr < i_obj):
        data = repair_json(raw[i_arr:], return_objects=True)
        if isinstance(data, list):
            return {list_key: data}, True
    if i_obj == -1:
        raise ValueError("No JSON object found in model output")
    e = raw.rfind("}")
    repaired = False
    try:
        obj = json.loads(raw[i_obj:e + 1])
    except Exception:
        repaired = True
        obj = repair_json(raw[i_obj:], return_objects=True)
    if not isinstance(obj, dict):
        raise ValueError("No JSON object found in model output")
    return obj, repaired


def parse_extraction(raw, max_topics):
    obj, repaired = extract_obj(raw, "topics")
    fixed = _fix_keys(obj, ["candidate_level", "topics"])
    normalized = set(fixed) != set(obj)
    obj = fixed

    lvl_raw = str(obj.get("candidate_level", "")).strip()
    lvl = LEVELS.get(lvl_raw.lower())
    if lvl is None:                       # الموديل نسي المستوى
        lvl, normalized = "Mid", True
    elif lvl != lvl_raw:
        normalized = True

    topics, seen = [], set()
    raw_topics = obj.get("topics")
    for tp in (raw_topics if isinstance(raw_topics, list) else []):
        if isinstance(tp, str):
            tp = {"name": tp}
        if not isinstance(tp, dict):
            continue
        tp = _fix_keys(tp, ["name", "type", "evidence"])
        name = str(tp.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        ty = TYPES.get(str(tp.get("type", "")).strip().lower(), "skill")
        topics.append({"name": name, "type": ty, "evidence": str(tp.get("evidence", "") or "").strip()})
    topics = topics[:max_topics]
    if not topics:
        raise ValueError("No topics found in the CV")

    result = extraction_parser.parse(json.dumps({"candidate_level": lvl, "topics": topics}, ensure_ascii=False))
    return result, {"repaired": repaired, "normalized": normalized}


def parse_topic_questions(raw, k):
    obj, repaired = extract_obj(raw, "questions")
    obj = _fix_keys(obj, ["questions"])
    normalized = False
    qs, seen = [], set()
    raw_qs = obj.get("questions")
    for q in (raw_qs if isinstance(raw_qs, list) else []):
        if not isinstance(q, dict):
            continue
        q = _fix_keys(q, ["question", "difficulty", "ideal_answer"])
        if not str(q.get("question", "")).strip() or not str(q.get("ideal_answer", "")).strip():
            continue                      # سؤال ناقص أو مقطوع
        key = re.sub(r"\W+", " ", str(q["question"]).lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        d_raw = str(q.get("difficulty", "")).strip()
        d = DIFFS.get(d_raw.lower())
        if d is None:
            d, normalized = "Medium", True
        elif d != d_raw:
            normalized = True
        qs.append({"question": str(q["question"]).strip(), "difficulty": d,
                   "ideal_answer": str(q["ideal_answer"]).strip()})
    if not qs:
        raise ValueError("No valid questions in model output")
    result = questions_parser.parse(json.dumps({"questions": qs[:k]}, ensure_ascii=False))
    return result, {"repaired": repaired, "normalized": normalized}


def drop_copies(questions, ctx):
    """يشيل الأسئلة اللي هي نفسها سؤال مرجع من الـ RAG."""
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    ref_qs = {norm(m) for c in ctx for m in re.findall(r"Question:\s*(.*?)\s*Ideal answer:", c)}
    return [q for q in questions if norm(q.question) not in ref_qs]


class PipelineError(Exception):
    def __init__(self, msg, raw=""):
        super().__init__(msg)
        self.raw = raw


def run_pipeline(cv, job, lang, k, max_topics, use_rag, llm, retrieve, on_stage=None):
    """CV -> مواضيع (مهارات/مشاريع) -> أسئلة لكل موضوع. llm و retrieve بيتبعتوا من برّه."""
    stage = on_stage or (lambda *a: None)
    stage("extract", 0, 1)
    raw1 = llm([(SYSTEM_EXTRACT, build_extract_user(cv, job, max_topics))], 600)[0]
    try:
        ext, info1 = parse_extraction(raw1, max_topics)
    except Exception as e:
        raise PipelineError(str(e), raw1)
    stage("extract", 1, 1)

    pairs, ctxs = [], []
    for tp in ext.topics:
        ctx = retrieve(f"{tp.name} {tp.evidence}".strip()) if use_rag else []
        ctxs.append(ctx)
        pairs.append((SYSTEM_Q, build_topic_user(cv, job, tp, ctx, k, lang)))
    raws = llm(pairs, 220 * k + 120, lambda d, n: stage("generate", d, n))

    sections, clean = [], int(not (info1["repaired"] or info1["normalized"]))
    for tp, ctx, raw in zip(ext.topics, ctxs, raws):
        try:
            tq, info = parse_topic_questions(raw, k)
            qs = drop_copies(tq.questions, ctx)
            if not qs:
                raise ValueError("All questions were copied from references")
            clean += int(not (info["repaired"] or info["normalized"]) and len(qs) == len(tq.questions))
            sections.append({"topic": tp, "questions": qs, "ctx": ctx, "ok": True})
        except Exception as e:
            sections.append({"topic": tp, "questions": [], "ctx": ctx, "ok": False,
                             "error": str(e), "raw": raw})
    stats = {"calls": 1 + len(pairs), "clean": clean,
             "failed": sum(1 for s in sections if not s["ok"])}
    return ext, sections, stats


# ---------- UI text ----------
TXT = {
    "ar": {
        "title": "مولّد أسئلة المقابلات الذكي",
        "sub": "ارفع السيرة الذاتية وسيستخرج النظام كل مهاراتك ومشاريعك، ثم يولّد أسئلة مقابلة لكل واحدة منها مع الإجابات النموذجية.",
        "settings": "⚙️ الإعدادات",
        "rag": "استخدام RAG (مراجع من قاعدة المعرفة)",
        "ft": "استخدام الموديل المدرَّب (Fine-tuned)",
        "ft_missing": "الموديل المدرَّب غير متاح بعد. يحتاج تنفيذ خطوة التدريب أولًا.",
        "k": "عدد الأسئلة لكل موضوع",
        "max_topics": "أقصى عدد مواضيع من السيرة",
        "job": "المسمى الوظيفي",
        "upload": "ارفع السيرة الذاتية (PDF)",
        "paste": "أو الصق نص السيرة الذاتية",
        "sample": "جرّب سيرة ذاتية تجريبية",
        "go": "ولّد الأسئلة",
        "empty": "ارفع ملف PDF أو الصق نص السيرة الذاتية أولًا.",
        "loading": "جاري تحميل الموديل (أول مرة تستغرق حوالي دقيقة)...",
        "extracting": "جاري قراءة السيرة الذاتية واستخراج المهارات والمشاريع...",
        "generating": "جاري توليد الأسئلة",
        "hint": "ستظهر النتائج هنا بعد الضغط على «ولّد الأسئلة».",
        "level": "المستوى المقدَّر",
        "topics": "مواضيع",
        "questions": "سؤال",
        "skill": "مهارة",
        "project": "مشروع",
        "from_cv": "من السيرة الذاتية",
        "answer": "الإجابة النموذجية",
        "refs": "المراجع المسترجعة (RAG)",
        "dl": "تحميل كل الأسئلة كملف نصي",
        "fail": "تعذّر قراءة السيرة الذاتية بشكل صحيح. جرّب مرة أخرى.",
        "topic_fail": "تعذّر توليد أسئلة لهذا الموضوع.",
        "raw": "المخرجات الخام",
        "q": "س",
        "valid": "خطوات سليمة من أول مرة",
        "fixed": "احتاجت إصلاحًا تلقائيًا",
        "read": "تمت قراءة",
        "chars": "حرفًا من السيرة",
        "time": "الزمن",
        "sec": "ثانية",
        "levels": {"Junior": "مبتدئ", "Mid": "متوسط", "Senior": "متقدم"},
        "diffs": {"Easy": "سهل", "Medium": "متوسط", "Hard": "صعب"},
    },
    "en": {
        "title": "AI Interview Question Generator",
        "sub": "Upload a CV: the system extracts all your skills and projects, then generates interview questions for each one with model answers.",
        "settings": "⚙️ Settings",
        "rag": "Use RAG (references from the knowledge base)",
        "ft": "Use fine-tuned model",
        "ft_missing": "Fine-tuned model not available yet. Run the training step first.",
        "k": "Questions per topic",
        "max_topics": "Max topics from the CV",
        "job": "Job title",
        "upload": "Upload CV (PDF)",
        "paste": "or paste the CV text",
        "sample": "Try a sample CV",
        "go": "Generate questions",
        "empty": "Upload a PDF or paste the CV text first.",
        "loading": "Loading the model (first run takes about a minute)...",
        "extracting": "Reading the CV and extracting skills and projects...",
        "generating": "Generating questions",
        "hint": "Results will appear here after you click “Generate questions”.",
        "level": "Estimated level",
        "topics": "topics",
        "questions": "questions",
        "skill": "Skill",
        "project": "Project",
        "from_cv": "From the CV",
        "answer": "Model answer",
        "refs": "Retrieved references (RAG)",
        "dl": "Download all questions as text",
        "fail": "Could not read the CV correctly. Please try again.",
        "topic_fail": "Could not generate questions for this topic.",
        "raw": "Raw output",
        "q": "Q",
        "valid": "steps valid on the first try",
        "fixed": "needed automatic repair",
        "read": "Read",
        "chars": "characters of the CV",
        "time": "Time",
        "sec": "s",
        "levels": {"Junior": "Junior", "Mid": "Mid", "Senior": "Senior"},
        "diffs": {"Easy": "Easy", "Medium": "Medium", "Hard": "Hard"},
    },
}


def to_text(ext, sections, lang, job):
    t = TXT[lang]
    lines = [t["title"], f"{job}", f"{t['level']}: {t['levels'][ext.candidate_level]}", "=" * 44, ""]
    n = 0
    for sec in sections:
        tp = sec["topic"]
        kind = t["project"] if tp.type == "project" else t["skill"]
        lines.append(f"## [{kind}] {tp.name}")
        if tp.evidence:
            lines.append(f"   {t['from_cv']}: {tp.evidence}")
        if not sec["ok"]:
            lines += [f"   ({t['topic_fail']})", ""]
            continue
        for q in sec["questions"]:
            n += 1
            lines += [f"{t['q']}{n}) [{t['diffs'][q.difficulty]}] {q.question}",
                      f"    {t['answer']}: {q.ideal_answer}"]
        lines.append("")
    return "\n".join(lines)


def rtl(text, cls="", lang="ar"):
    d = "rtl" if lang == "ar" else "ltr"
    return f'<div dir="{d}" class="{cls}">{html.escape(text)}</div>'
# ===== HELPERS END =====

SAMPLE_CV = (
    "Ahmed Ali. Backend Developer with 3 years of experience. Skills: Python, Django, PostgreSQL, Docker, "
    "REST APIs, Redis, Git. Experience: Backend developer at a fintech startup (2022-2025). "
    "Projects: (1) E-commerce platform with 50k products and 2k daily orders, built with Django and PostgreSQL, "
    "reduced page load time by 40% using Redis caching. (2) Payment API with JWT authentication and rate limiting, "
    "containerised with Docker and deployed on AWS. (3) Sales dashboard using Pandas and Matplotlib for the "
    "management team.")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
.hero, .qtext, .atext, .topic, .stMarkdown p, label p, textarea, input { font-family: 'Cairo', sans-serif; }
.hero {background: linear-gradient(135deg,#4f46e5,#7c3aed 60%,#06b6d4); padding: 26px 32px;
       border-radius: 18px; color: #fff; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff; padding: 0;}
.hero p {margin: 6px 0 0; opacity: .93; font-size: 1.02rem;}
.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: .82rem;
       font-weight: 600; margin: 0 0 0 6px;}
.c-easy {background: #dcfce7; color: #166534;}
.c-medium {background: #fef3c7; color: #92400e;}
.c-hard {background: #fee2e2; color: #991b1b;}
.c-skill {background: #e0e7ff; color: #3730a3;}
.c-project {background: #cffafe; color: #155e75;}
.c-level {background: #ede9fe; color: #5b21b6; font-size: 1rem; padding: 6px 18px;}
.c-mode {background: #e2e8f0; color: #334155;}
.topic {font-size: 1.25rem; font-weight: 700;}
.evid {color: #64748b; font-size: .88rem; margin-top: 4px;}
.qtext {font-size: 1.05rem; font-weight: 600; line-height: 1.9; margin: 8px 0 2px;}
.atext {line-height: 1.95;}
</style>
"""


def card():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


# ---------- Loaders (بيتحملوا مرة واحدة) ----------
@st.cache_resource(show_spinner=False)
def load_retriever():
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"})
    db = Chroma(persist_directory="chroma_db", embedding_function=emb)
    return db.as_retriever(search_kwargs={"k": 2})


@st.cache_resource(show_spinner=False)
def load_llm():
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
    if os.path.isdir(ADAPTER):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model


def make_llm(tok, model, has_adapter, use_ft):
    def llm(pairs, max_new_tokens, on_batch=None):
        outs = []
        for b in range(0, len(pairs), BATCH):
            chunk = pairs[b:b + BATCH]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True) for s, u in chunk]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            cm = model.disable_adapter() if (has_adapter and not use_ft) else nullcontext()
            with cm, torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                     temperature=0.4, top_p=0.9, pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            if on_batch:
                on_batch(min(b + BATCH, len(pairs)), len(pairs))
        return outs
    return llm


# ---------- Sidebar ----------
has_adapter = os.path.isdir(ADAPTER)
with st.sidebar:
    lang_choice = st.radio("اللغة / Language", ["العربية", "English"], horizontal=True)
    lang = "ar" if lang_choice == "العربية" else "en"
    t = TXT[lang]
    st.header(t["settings"])
    use_rag = st.toggle(t["rag"], value=True)
    use_ft = st.toggle(t["ft"], value=has_adapter, disabled=not has_adapter)
    if not has_adapter:
        st.caption("🔒 " + t["ft_missing"])
    k = st.slider(t["k"], 2, 5, 3)
    max_topics = st.slider(t["max_topics"], 3, 12, 8)

DIR = "rtl" if lang == "ar" else "ltr"
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(f'<div class="hero" dir="{DIR}"><h1>🎯 {t["title"]}</h1><p>{t["sub"]}</p></div>',
            unsafe_allow_html=True)


def load_sample():
    st.session_state["cv_text"] = SAMPLE_CV


col_in, col_out = st.columns([1, 1.5], gap="large")

with col_in:
    with card():
        job = st.text_input(t["job"], "Backend Developer (Django)")
        pdf = st.file_uploader(t["upload"], type=["pdf"])
        st.text_area(t["paste"], height=260, key="cv_text")
        st.button(t["sample"], on_click=load_sample)
        run = st.button(t["go"], type="primary", use_container_width=True)

cv_text = st.session_state.get("cv_text", "")
if pdf is not None:
    cv_text = "\n".join((p.extract_text() or "") for p in PdfReader(pdf).pages)

with col_out:
    if run:
        cv = " ".join(cv_text.split())[:CV_CHARS]
        if not cv:
            st.warning(t["empty"])
        else:
            with st.spinner(t["loading"]):
                retriever = load_retriever()
                tok, model = load_llm()
            llm = make_llm(tok, model, has_adapter, use_ft)
            retrieve = lambda q: [d.page_content for d in retriever.invoke(q)]
            bar = st.progress(0.0, text=t["extracting"])

            def on_stage(name, done, total):
                if name == "extract":
                    bar.progress(0.05 if done else 0.0, text=t["extracting"])
                else:
                    bar.progress(0.1 + 0.9 * done / max(total, 1), text=f'{t["generating"]} ({done}/{total})')

            t0 = time.time()
            try:
                ext, sections, stats = run_pipeline(cv, job, lang, k, max_topics, use_rag,
                                                    llm, retrieve, on_stage)
                st.session_state["out"] = {"ext": ext, "sections": sections, "stats": stats,
                                           "secs": time.time() - t0, "job": job, "rag": use_rag,
                                           "ft": bool(use_ft and has_adapter), "cv_len": len(cv)}
            except PipelineError as ex:
                st.session_state["out"] = {"error": str(ex), "raw": ex.raw}
            except Exception as ex:
                st.session_state["out"] = {"error": str(ex), "raw": ""}
            bar.empty()

    res = st.session_state.get("out")
    if res is None:
        st.info(t["hint"])
    elif "error" in res:
        st.error(t["fail"] + "\n\n" + res["error"][:300])
        if res["raw"]:
            with st.expander(t["raw"]):
                st.code(res["raw"])
    else:
        ext, sections, stats = res["ext"], res["sections"], res["stats"]
        total_q = sum(len(s["questions"]) for s in sections)
        mode = f'RAG {"✔" if res["rag"] else "✘"} · Fine-tuned {"✔" if res["ft"] else "✘"}'
        st.markdown(
            f'<div dir="{DIR}"><span class="chip c-level">{t["level"]}: {t["levels"][ext.candidate_level]}</span>'
            f'<span class="chip c-mode">{len(sections)} {t["topics"]} · {total_q} {t["questions"]}</span>'
            f'<span class="chip c-mode">{mode}</span></div>', unsafe_allow_html=True)
        st.caption(f'{t["read"]} {res["cv_len"]} {t["chars"]}')

        n = 0
        for sec in sections:
            tp = sec["topic"]
            with card():
                kind = t["project"] if tp.type == "project" else t["skill"]
                ev = (f'<div class="evid">{t["from_cv"]}: {html.escape(tp.evidence)}</div>'
                      if tp.evidence else "")
                st.markdown(
                    f'<div dir="{DIR}"><span class="chip c-{tp.type}">{kind}</span>'
                    f'<span class="topic">{html.escape(tp.name)}</span>{ev}</div>',
                    unsafe_allow_html=True)
                if not sec["ok"]:
                    st.warning(t["topic_fail"])
                    continue
                for q in sec["questions"]:
                    n += 1
                    st.markdown(
                        f'<div dir="{DIR}"><span class="chip c-{q.difficulty.lower()}">{t["diffs"][q.difficulty]}</span>'
                        f'<div class="qtext">{t["q"]}{n}. {html.escape(q.question)}</div></div>',
                        unsafe_allow_html=True)
                    with st.expander(t["answer"]):
                        st.markdown(rtl(q.ideal_answer, "atext", lang), unsafe_allow_html=True)

        st.download_button(t["dl"], to_text(ext, sections, lang, res["job"]),
                           file_name="interview_questions.txt", mime="text/plain",
                           use_container_width=True)
        st.caption(f'{stats["clean"]}/{stats["calls"]} {t["valid"]} · {t["time"]}: {res["secs"]:.0f} {t["sec"]}')
        refs = [(s["topic"].name, s["ctx"]) for s in sections if s["ctx"]]
        if refs:
            with st.expander(t["refs"]):
                for name, ctx in refs:
                    st.markdown(f"**{name}**")
                    for c in ctx:
                        st.write("•", c[:200])
