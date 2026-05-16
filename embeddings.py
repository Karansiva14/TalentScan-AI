"""
embeddings.py — Build and query a FAISS vector store from resume text chunks.
Uses sentence-transformers for local CPU-friendly embeddings.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Lightweight model — fast on CPU, strong semantic quality
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of text strings into L2-normalised embedding vectors.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), dtype float32.
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # L2 normalise for cosine similarity via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)
    return embeddings.astype(np.float32)


def build_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """
    Embed chunks and build an in-memory FAISS IndexFlatIP (inner product / cosine).

    Args:
        chunks: List of text chunks from the resume.

    Returns:
        (faiss_index, embeddings_matrix)
    """
    if not chunks:
        raise ValueError("chunks list is empty — cannot build FAISS index.")

    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def retrieve_top_k(
    query: str,
    chunks: List[str],
    index: faiss.IndexFlatIP,
    k: int = 5,
) -> List[str]:
    """
    Embed a query and retrieve the top-k most relevant resume chunks.

    Args:
        query:  The job description or query string.
        chunks: Original text chunks (parallel to index).
        index:  Built FAISS index.
        k:      Number of top chunks to retrieve.

    Returns:
        List of top-k chunk strings, ranked by cosine similarity.
    """
    k = min(k, len(chunks))
    query_vec = embed_texts([query])  # shape (1, dim)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
