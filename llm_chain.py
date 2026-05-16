"""
llm_chain.py — LangChain + Groq API chain for structured job-fit analysis.
Produces ATS score, matched skills, skill gaps, and improvement suggestions
in a single inference pass, output as validated JSON.
"""

import json
import os
import re
from typing import Any

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ── Model config ─────────────────────────────────────────────────────────────
_MODEL_ID = "llama3-70b-8192"
_TEMPERATURE = 0.2
_MAX_TOKENS = 1800

# ── Prompt ───────────────────────────────────────────────────────────────────
_SYSTEM = """You are TalentScan AI — an expert Applicant Tracking System (ATS) and
HR analyst. You receive semantically relevant excerpts from a candidate's resume
alongside a job description. Your task is to return a structured JSON analysis
with ZERO additional commentary, preamble, or markdown code fences.

Output ONLY valid JSON matching this schema exactly:
{{
  "ats_score": <integer 0-100>,
  "score_rationale": "<2-sentence explanation of the score>",
  "matched_skills": ["<skill>", ...],
  "missing_skills": ["<skill>", ...],
  "experience_fit": "<Strong | Moderate | Weak> — <one sentence>",
  "education_fit": "<Strong | Moderate | Weak> — <one sentence>",
  "improvement_suggestions": [
    "<actionable suggestion 1>",
    "<actionable suggestion 2>",
    "<actionable suggestion 3>"
  ],
  "hiring_recommendation": "<Highly Recommended | Recommended | Maybe | Not Recommended>",
  "summary": "<3-sentence overall candidate summary for an HR reviewer>"
}}"""

_HUMAN = """JOB DESCRIPTION:
{job_description}

---
RESUME EXCERPTS (top semantically matched chunks):
{resume_chunks}

---
Return the JSON analysis now."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])


def _build_chain(api_key: str):
    """Construct the LangChain runnable chain."""
    llm = ChatGroq(
        api_key=api_key,
        model=_MODEL_ID,
        temperature=_TEMPERATURE,
        max_tokens=_MAX_TOKENS,
    )
    return _prompt | llm | StrOutputParser()


def _extract_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON robustly."""
    # Remove ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Find first { ... } block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:500]}")
    return json.loads(match.group())


def analyze_resume(
    job_description: str,
    resume_chunks: list[str],
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Run the full LLM analysis pipeline.

    Args:
        job_description: Raw text of the target job description.
        resume_chunks:   Top-k semantically retrieved resume chunks.
        api_key:         Groq API key. Falls back to GROQ_API_KEY env var.

    Returns:
        Parsed analysis dict matching the schema above.
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Groq API key not found. Set GROQ_API_KEY or pass api_key argument."
        )

    chain = _build_chain(key)

    chunks_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{c}" for i, c in enumerate(resume_chunks)
    )

    raw_output = chain.invoke({
        "job_description": job_description.strip(),
        "resume_chunks": chunks_block,
    })

    return _extract_json(raw_output)
