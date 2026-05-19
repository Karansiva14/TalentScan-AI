# TalentScan AI 🎯

> **AI-powered resume screening** — RAG pipeline that ingests candidate PDFs, scores job-fit, and delivers structured analysis in under 3 seconds.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![LLaMA 3](https://img.shields.io/badge/LLaMA_3-70B-purple?style=flat-square)](https://ai.meta.com/llama/)
[![Groq](https://img.shields.io/badge/Groq-Accelerated-orange?style=flat-square)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-green?style=flat-square)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## What it does

HR teams spend hours manually screening resumes. TalentScan AI reduces that to **under 3 seconds per candidate** — with zero data storage and full privacy.

Upload a resume PDF + paste a job description → get back:

- **ATS compatibility score** (0–100)
- **Matched skills** against the JD
- **Skill gap report** — what's missing
- **Improvement suggestions** — actionable, specific
- All output structured as **JSON** for downstream processing

---

## Architecture

```
Candidate PDF
      │
      ▼
┌─────────────────┐
│  PyMuPDF Parser │  ← Extract raw text from PDF
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Text Chunker        │  ← Split into semantic chunks
│  + Embedder          │  ← Sentence-transformer embeddings
└────────┬────────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │  ← Store & index all chunks
│  Store          │
└────────┬────────┘
         │  top-k semantic retrieval
         ▼
┌──────────────────────────┐
│  LLaMA 3 70B via Groq    │  ← Single inference pass
│  (hardware-accelerated)  │  ← Prompt: resume chunks + JD
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────┐
│  Structured JSON     │  ← ATS score, skills, gaps,
│  Output              │     suggestions
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Streamlit Dashboard │  ← HR-facing UI
└──────────────────────┘
```

**Key design choices:**
- **Groq inference** for hardware-accelerated LLM calls — achieves sub-3s end-to-end latency
- **FAISS** for in-memory vector search — no external vector DB dependency
- **Single inference pass** — ATS score + skills + gaps + suggestions in one LLM call, structured as JSON
- **Zero data storage** — resumes are processed in memory only, never persisted

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3 70B (via Groq API) |
| Orchestration | LangChain |
| Vector Store | FAISS |
| PDF Parsing | PyMuPDF |
| Embeddings | Sentence Transformers |
| UI | Streamlit |
| Language | Python 3.10+ |

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repo
git clone https://github.com/karansiva14/talentscan-ai.git
cd talentscan-ai

# Install dependencies
pip install -r requirements.txt

# Set up your Groq API key
cp .env.example .env
# Add your key: GROQ_API_KEY=your_key_here
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. Upload a candidate's resume as a PDF
2. Paste the job description into the text field
3. Click **Analyze**
4. Get structured results in < 3 seconds

### Sample Output (JSON)

```json
{
  "ats_score": 82,
  "matched_skills": [
    "Python", "Machine Learning", "LangChain", "RAG", "FAISS", "NLP"
  ],
  "skill_gaps": [
    "Docker / containerization",
    "MLOps / model serving (FastAPI, BentoML)",
    "Cloud deployment (AWS SageMaker / GCP Vertex AI)"
  ],
  "improvement_suggestions": [
    "Add a Docker container for the Streamlit app to demonstrate deployment readiness",
    "Quantify project impact with specific metrics (latency, accuracy, dataset size)",
    "Include cloud platform experience — most production ML roles require AWS or GCP"
  ],
  "summary": "Strong candidate for junior AI/ML roles with solid RAG and LLM experience. Skill gaps are primarily in production deployment and MLOps tooling."
}
```

---

## Performance

| Metric | Value |
|---|---|
| End-to-end latency | < 3 seconds |
| LLM | LLaMA 3 70B |
| Inference backend | Groq (hardware-accelerated) |
| Data storage | Zero — fully in-memory |
| Supported input | PDF resumes |

---

## Project Structure

```
talentscan-ai/
├── app.py                  # Streamlit UI entry point
├── pipeline/
│   ├── parser.py           # PyMuPDF PDF extractor
│   ├── chunker.py          # Text chunker + embedder
│   ├── vector_store.py     # FAISS index builder & retriever
│   └── analyzer.py         # LLaMA 3 prompt + JSON parser
├── prompts/
│   └── analysis_prompt.txt # System + user prompt templates
├── requirements.txt
├── .env.example
└── README.md
```

---

## Roadmap

- [ ] Multi-resume batch screening (upload 10+ PDFs at once)
- [ ] Job description scraper (paste a LinkedIn URL instead of text)
- [ ] Candidate ranking / leaderboard across multiple resumes
- [ ] Swap-ready model support (any GGUF-compatible model)
- [ ] REST API endpoint (FastAPI wrapper)
- [ ] Export report as PDF

---

## About

Built by **Karan S H** — AI/ML Engineer with an IIT Minor in Artificial Intelligence & Data Science.

- LinkedIn: [linkedin.com/in/karan1414](https://linkedin.com/in/karan1414)
- GitHub: [github.com/karansiva14](https://github.com/karansiva14)
- Email: karansmiley14@gmail.com

---

## License

MIT License — free to use, modify, and distribute. See [LICENSE](LICENSE) for details.
