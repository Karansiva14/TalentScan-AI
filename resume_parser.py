"""
resume_parser.py — Extract and chunk text from candidate PDF resumes using PyMuPDF.
"""

import fitz  # PyMuPDF
import re
from typing import List


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract full plain text from a PDF (in-memory bytes).
    Returns cleaned text string.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
    doc.close()
    raw = "\n".join(pages_text)
    return _clean_text(raw)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split resume text into overlapping word-based chunks for FAISS indexing.

    Args:
        text:       Full resume text.
        chunk_size: Target words per chunk.
        overlap:    Words of overlap between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def parse_resume(pdf_bytes: bytes) -> dict:
    """
    Full pipeline: PDF bytes → cleaned text → chunks.

    Returns:
        {
            "full_text": str,
            "chunks":    List[str],
            "num_pages": int,
            "num_chunks": int,
        }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = doc.page_count
    doc.close()

    full_text = extract_text_from_pdf(pdf_bytes)
    chunks = chunk_text(full_text)

    return {
        "full_text": full_text,
        "chunks": chunks,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
    }
