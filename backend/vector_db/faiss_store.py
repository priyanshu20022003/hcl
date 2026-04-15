"""
FAISS vector store wrapper.

Wraps a FAISS IndexFlatIP (inner-product / cosine similarity on normalised
vectors) with parallel metadata storage for document chunks.
"""

import json
import threading
from pathlib import Path

import faiss
import numpy as np

from .embeddings import EMBEDDING_DIM


class FaissStore:
    """In-memory FAISS index with associated chunk metadata."""

    def __init__(self):
        # Inner-product on L2-normalised vectors == cosine similarity
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.metadata: list[dict] = []  # parallel list: one entry per vector
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def add(self, embeddings: np.ndarray, metadata_list: list[dict]):
        """Add vectors and their metadata to the index.

        Args:
            embeddings: (N, dim) float32 array of L2-normalised vectors.
            metadata_list: list of dicts, one per vector, e.g.
                {"document_id": "DOC-001", "chunk_index": 0,
                 "chunk_text": "...", "title": "..."}
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("embeddings and metadata_list must have the same length")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        with self._lock:
            self.index.add(embeddings)
            self.metadata.extend(metadata_list)

    def add_single(self, embedding: np.ndarray, metadata: dict):
        """Add a single vector and its metadata to the index."""
        emb = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        with self._lock:
            self.index.add(emb)
            self.metadata.append(metadata)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top-k nearest chunks with similarity scores.

        Args:
            query_embedding: (dim,) float32 normalised vector.
            top_k: number of results to return.

        Returns:
            List of dicts: each metadata dict augmented with a 'score' key.
        """
        if self.index.ntotal == 0:
            return []
        top_k = min(top_k, self.index.ntotal)
        query = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(query, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, directory: str | Path):
        """Save the FAISS index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        with open(directory / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"[faiss_store] Saved {self.index.ntotal} vectors to {directory}")

    def load(self, directory: str | Path):
        """Load a previously saved FAISS index and metadata from disk."""
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[faiss_store] Loaded {self.index.ntotal} vectors from {directory}")

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
