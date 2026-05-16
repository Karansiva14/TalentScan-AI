"""
app.py — TalentScan AI · Streamlit Dashboard
LLM-powered resume analyser: PDF → RAG → LLaMA 3 → structured ATS report.
"""

import time
import streamlit as st

from resume_parser import parse_resume
from embeddings import build_faiss_index, retrieve_top_k
from llm_chain import analyze_resume

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TalentScan AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0f14;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1e2330;
}

/* Cards */
.card {
    background: #151820;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card h4 { margin-top: 0; color: #7dd3fc; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* Score ring */
.score-ring {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center;
    width: 130px; height: 130px;
    border-radius: 50%;
    font-size: 2.2rem; font-weight: 800;
    margin: 0 auto 0.6rem;
    border: 6px solid;
}
.score-high  { border-color: #22c55e; color: #22c55e; }
.score-mid   { border-color: #f59e0b; color: #f59e0b; }
.score-low   { border-color: #ef4444; color: #ef4444; }

/* Pills */
.pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem 0.15rem;
}
.pill-green { background: #14532d55; border: 1px solid #22c55e; color: #4ade80; }
.pill-red   { background: #4c000055; border: 1px solid #ef4444; color: #f87171; }

/* Rec badge */
.rec-badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 0.9rem;
}
.rec-high   { background: #14532d55; border: 1px solid #22c55e; color: #4ade80; }
.rec-mid    { background: #78350f55; border: 1px solid #f59e0b; color: #fcd34d; }
.rec-maybe  { background: #1e3a5f55; border: 1px solid #60a5fa; color: #93c5fd; }
.rec-low    { background: #4c000055; border: 1px solid #ef4444; color: #f87171; }

/* Suggestions */
.suggestion-item {
    padding: 0.55rem 0.9rem;
    margin: 0.35rem 0;
    border-left: 3px solid #6366f1;
    background: #1a1d2e;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    color: #cbd5e1;
}

/* Metric row */
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; }
.metric-box {
    flex:1; min-width:140px;
    background:#151820; border:1px solid #1e2330;
    border-radius:10px; padding:0.9rem 1rem;
}
.metric-box .label { font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:0.07em; }
.metric-box .value { font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-top:0.2rem; }

/* Latency */
.latency-bar {
    font-size:0.78rem; color:#475569;
    text-align:right; margin-top:0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TalentScan AI")
    st.markdown("<p style='color:#64748b;font-size:0.82rem;'>LLM·RAG·ATS Analyser — Powered by LLaMA 3 70B via Groq</p>", unsafe_allow_html=True)
    st.divider()

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at console.groq.com",
    )

    top_k = st.slider("Top-K chunks retrieved", min_value=3, max_value=10, value=5,
                      help="Number of semantically relevant resume chunks sent to LLM.")

    st.divider()
    st.markdown("""
<div style='font-size:0.75rem;color:#475569;'>
<b style='color:#94a3b8;'>Stack</b><br>
Groq · LLaMA 3 70B · FAISS<br>
LangChain · Streamlit · PyMuPDF<br>
sentence-transformers<br><br>
<b style='color:#94a3b8;'>Privacy</b><br>
Zero data stored. All processing<br>in-session memory only.
</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e2e8f0;margin-bottom:0.2rem;'>TalentScan <span style='color:#6366f1;'>AI</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#475569;margin-top:0;'>Resume → RAG → LLM → ATS Report · Sub-3s inference</p>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📄 Upload Resume (PDF)")
    uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf:
        st.success(f"✓ {uploaded_pdf.name}")

with col2:
    st.markdown("#### 💼 Job Description")
    job_desc = st.text_area(
        "",
        height=180,
        placeholder="Paste the full job description here — role requirements, skills, responsibilities...",
        label_visibility="collapsed",
    )

st.markdown("")
run_col, _ = st.columns([1, 3])
with run_col:
    analyse_btn = st.button("⚡ Analyse Resume", use_container_width=True, type="primary")

# ── Pipeline ──────────────────────────────────────────────────────────────────
if analyse_btn:
    errors = []
    if not groq_key:
        errors.append("⚠️ Enter your Groq API key in the sidebar.")
    if not uploaded_pdf:
        errors.append("⚠️ Upload a PDF resume.")
    if not job_desc.strip():
        errors.append("⚠️ Paste a job description.")

    if errors:
        for e in errors:
            st.warning(e)
        st.stop()

    t0 = time.perf_counter()

    with st.status("Running TalentScan AI pipeline…", expanded=True) as status:
        st.write("📄 Parsing PDF resume…")
        pdf_bytes = uploaded_pdf.read()
        parsed = parse_resume(pdf_bytes)
        st.write(f"   → {parsed['num_pages']} page(s) · {parsed['num_chunks']} chunks extracted")

        st.write("🔗 Building FAISS vector index…")
        index, _ = build_faiss_index(parsed["chunks"])
        st.write(f"   → Indexed {len(parsed['chunks'])} chunks (dim=384)")

        st.write(f"🎯 Retrieving top-{top_k} semantically relevant chunks…")
        top_chunks = retrieve_top_k(job_desc, parsed["chunks"], index, k=top_k)
        st.write(f"   → {len(top_chunks)} chunks retrieved for LLM context")

        st.write("🤖 Querying LLaMA 3 70B via Groq API…")
        result = analyze_resume(job_desc, top_chunks, api_key=groq_key)
        st.write("   → Structured JSON analysis received ✓")

        status.update(label="✅ Analysis complete!", state="complete")

    latency = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("## 📊 ATS Analysis Report")

    # Score
    score = result.get("ats_score", 0)
    ring_class = "score-high" if score >= 75 else ("score-mid" if score >= 50 else "score-low")
    rec = result.get("hiring_recommendation", "—")
    rec_class = (
        "rec-high" if "Highly" in rec else
        "rec-mid"  if rec == "Recommended" else
        "rec-maybe" if "Maybe" in rec else "rec-low"
    )

    s1, s2, s3 = st.columns([1, 1.8, 1.2])

    with s1:
        st.markdown(f"""
<div class='card' style='text-align:center;'>
  <h4>ATS Score</h4>
  <div class='score-ring {ring_class}'>{score}</div>
  <div style='font-size:0.78rem;color:#64748b;'>out of 100</div>
</div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
<div class='card'>
  <h4>Score Rationale</h4>
  <p style='font-size:0.88rem;color:#cbd5e1;margin:0;'>{result.get("score_rationale","—")}</p>
</div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
<div class='card' style='text-align:center;'>
  <h4>Hiring Recommendation</h4>
  <div style='margin-top:0.8rem;'>
    <span class='rec-badge {rec_class}'>{rec}</span>
  </div>
</div>""", unsafe_allow_html=True)

    # Fit metrics
    st.markdown(f"""
<div class='metric-row' style='margin-bottom:1rem;'>
  <div class='metric-box'><div class='label'>Experience Fit</div><div class='value'>{result.get("experience_fit","—")}</div></div>
  <div class='metric-box'><div class='label'>Education Fit</div><div class='value'>{result.get("education_fit","—")}</div></div>
  <div class='metric-box'><div class='label'>Chunks → LLM</div><div class='value'>{len(top_chunks)} / {parsed["num_chunks"]}</div></div>
  <div class='metric-box'><div class='label'>Pages Parsed</div><div class='value'>{parsed["num_pages"]}</div></div>
</div>""", unsafe_allow_html=True)

    # Skills columns
    c1, c2 = st.columns(2)
    with c1:
        matched = result.get("matched_skills", [])
        pills = " ".join(f"<span class='pill pill-green'>✓ {s}</span>" for s in matched)
        st.markdown(f"""
<div class='card'>
  <h4>✅ Matched Skills ({len(matched)})</h4>
  <div style='margin-top:0.5rem;'>{pills if pills else "<span style='color:#475569'>None identified</span>"}</div>
</div>""", unsafe_allow_html=True)

    with c2:
        missing = result.get("missing_skills", [])
        pills_m = " ".join(f"<span class='pill pill-red'>✗ {s}</span>" for s in missing)
        st.markdown(f"""
<div class='card'>
  <h4>❌ Skill Gaps ({len(missing)})</h4>
  <div style='margin-top:0.5rem;'>{pills_m if pills_m else "<span style='color:#475569'>None identified</span>"}</div>
</div>""", unsafe_allow_html=True)

    # Suggestions
    suggestions = result.get("improvement_suggestions", [])
    if suggestions:
        items = "".join(f"<div class='suggestion-item'>💡 {s}</div>" for s in suggestions)
        st.markdown(f"""
<div class='card'>
  <h4>🚀 Improvement Suggestions</h4>
  <div style='margin-top:0.5rem;'>{items}</div>
</div>""", unsafe_allow_html=True)

    # Summary
    st.markdown(f"""
<div class='card'>
  <h4>📝 Candidate Summary</h4>
  <p style='font-size:0.9rem;color:#cbd5e1;margin:0;line-height:1.6;'>{result.get("summary","—")}</p>
</div>""", unsafe_allow_html=True)

    # Latency
    st.markdown(f"<div class='latency-bar'>⚡ End-to-end latency: <b>{latency:.2f}s</b> &nbsp;|&nbsp; Model: LLaMA 3 70B via Groq &nbsp;|&nbsp; Privacy: zero storage</div>",
                unsafe_allow_html=True)

    # Raw JSON expander
    with st.expander("🗂 View raw JSON output"):
        st.json(result)
