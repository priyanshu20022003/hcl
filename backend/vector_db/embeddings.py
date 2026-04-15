"""
Embedding utility using sentence-transformers.

Uses the lightweight 'all-MiniLM-L6-v2' model (~80MB) to generate
384-dimensional embeddings for document chunks and queries.
"""

import numpy as np

_model = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _load_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[embeddings] Loading model '{MODEL_NAME}' ...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[embeddings] Model loaded.")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Return a single embedding vector for the given text."""
    model = _load_model()
    return model.encode(text, normalize_embeddings=True)


def get_embeddings(texts: list[str]) -> np.ndarray:
    """Return a batch of embedding vectors (N x dim) for a list of texts."""
    model = _load_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 20)
