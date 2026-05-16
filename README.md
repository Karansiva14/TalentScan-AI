# 🔍 TalentScan AI

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=langchain&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203%2070B-F55036?style=flat)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=flat)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> **LLM-powered resume analyser built on a full RAG pipeline.**  
> PDF → chunk → embed → FAISS retrieve → LLaMA 3 70B → structured ATS JSON report — all in **under 3 seconds**.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📄 PDF Parsing | PyMuPDF extracts & cleans text from any resume layout |
| 🔗 RAG Pipeline | Word-level chunking → sentence-transformer embeddings → FAISS cosine retrieval |
| 🤖 LLM Analysis | LLaMA 3 70B via Groq API — structured JSON output in a single inference pass |
| 📊 ATS Report | Score, matched skills, skill gaps, fit assessment, improvement suggestions |
| ⚡ Sub-3s Latency | Groq hardware-accelerated inference; typical end-to-end < 3 seconds |
| 🔒 Privacy First | Zero data storage — all processing happens in-session memory |
| 🎛 Interactive UI | Dark-themed Streamlit dashboard with skill pills, score ring, rec badge |

---

## 🏗 Architecture

```
PDF Resume
    │
    ▼
resume_parser.py   ── PyMuPDF ──► full text ──► word-based chunks (400w, 80w overlap)
    │
    ▼
embeddings.py      ── sentence-transformers (all-MiniLM-L6-v2)
                   ── FAISS IndexFlatIP (cosine via L2-normalised inner product)
                   ── retrieve top-k chunks matching job description query
    │
    ▼
llm_chain.py       ── LangChain ChatPromptTemplate
                   ── Groq API → LLaMA 3 70B (temp=0.2, max_tokens=1800)
                   ── Single-pass structured JSON output
    │
    ▼
app.py             ── Streamlit dark-themed dashboard
                   ── Score ring · skill pills · rec badge · latency footer
```

---

## 📁 Project Structure

```
TalentScan-AI/
├── app.py              # Streamlit dashboard (main entry point)
├── resume_parser.py    # PDF text extraction & chunking (PyMuPDF)
├── embeddings.py       # FAISS vector index build & retrieval
├── llm_chain.py        # LangChain + Groq LLM chain (structured JSON)
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/karansiva14/TalentScan-AI.git
cd TalentScan-AI

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
cp .env.example .env
# Edit .env — paste your key from https://console.groq.com
```

Or enter the key directly in the Streamlit sidebar at runtime (no `.env` needed).

### 3. Run

```bash
streamlit run app.py
```

Open **http://localhost:8501** → upload a PDF resume → paste a job description → click **Analyse Resume**.

---

## 🌐 Deploy on Hugging Face Spaces

1. Create a new Space → SDK: **Streamlit**
2. Push this repo to the Space
3. Add `GROQ_API_KEY` as a **Secret** in Space Settings
4. Done — public URL in < 2 minutes

---

## 📦 LLM Output Schema

```json
{
  "ats_score": 82,
  "score_rationale": "Strong alignment on core ML stack. Minor gaps in cloud tooling.",
  "matched_skills": ["Python", "TensorFlow", "LangChain", "FAISS"],
  "missing_skills": ["AWS SageMaker", "Docker"],
  "experience_fit": "Strong — internship and projects directly match role requirements.",
  "education_fit": "Strong — B.E. ECE + IIT Minor in AI/DS exceeds minimum requirements.",
  "improvement_suggestions": [
    "Add Docker and containerisation experience to resume.",
    "Quantify model performance metrics (accuracy, latency) in project bullets.",
    "Include a cloud deployment section highlighting Hugging Face Spaces deployment."
  ],
  "hiring_recommendation": "Highly Recommended",
  "summary": "Strong fresher candidate with hands-on LLM/RAG experience..."
}
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | LLaMA 3 70B via Groq API |
| Orchestration | LangChain 0.2 |
| Vector DB | FAISS (CPU, cosine similarity) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF Parsing | PyMuPDF (fitz) |
| UI | Streamlit 1.35+ |
| Language | Python 3.11+ |

---

## 👤 Author

**Karan S H** · [LinkedIn](https://linkedin.com/in/karan1414) · [GitHub](https://github.com/karansiva14)  
B.E. ECE · IIT Minor in AI & Data Science

---

## 📄 License

MIT — free to use, modify, and distribute.
