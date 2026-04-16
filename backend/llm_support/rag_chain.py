"""
RAG Chain — orchestrates retrieval from FAISS and answer generation via Sarvam.

Pipeline:
  1. Embed the user query
  2. Search FAISS for top-k relevant chunks
  3. Build a grounded context prompt
  4. Call Sarvam LLM for the final answer
  5. Optionally translate to Hindi
"""

import re
import sqlite3
import numpy as np
from pathlib import Path

from ..vector_db.embeddings import get_embedding
from ..vector_db.faiss_store import FaissStore
from .sarvam_client import SarvamClient


class RAGChain:
    """End-to-end RAG pipeline: FAISS retrieval → Sarvam LLM generation."""

    def __init__(self, faiss_store: FaissStore, sarvam: SarvamClient, db_path: Path = None, experience_store: FaissStore = None):
        self.store = faiss_store
        self.sarvam = sarvam
        self.db_path = db_path
        self.experience_store = experience_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        preferred_language: str = "auto",
        top_k: int = 5,
        include_voice: bool = False,
    ) -> dict:
        """Run the full RAG pipeline and return a structured response."""
        detected_lang = self._detect_language(query, preferred_language)

        # 1. Retrieve relevant chunks from FAISS
        query_embedding = get_embedding(query)
        results = self.store.search(query_embedding, top_k=top_k)
        
        # 1.5 Augment results with dynamic utility scores from DB
        results = self._augment_with_utility(results)

        # 2. Retrieve past lessons (Semantic Memory from Experience Brain)
        lessons = self._get_past_lessons(query_embedding)
        lessons_string = ""
        if lessons:
            lessons_string = "\n".join([f"- Verified Correction: {l}" for l in lessons])

        if not results and not lessons:
            return self._no_info_response(query, detected_lang)

        # 3. De-duplicate by document and build context
        sources = self._build_sources(results)
        context_string = self._format_context(results)

        # 4. Generate answer via Sarvam LLM
        system_prompt = (
            "You are PolicySarthi, a hospital policy assistant using retrieval-augmented generation. "
            "Answer strictly based on the provided context. If the answer is not in the context, "
            "say that clearly. Do not invent policies, steps, or documents.\n\n"
        )
        
        if lessons_string:
            system_prompt += (
                "IMPORTANT: You have previously learned the following corrections for similar queries. "
                "Incorporate these verified corrections into your response:\n"
                f"{lessons_string}\n\n"
            )
            
        system_prompt += "Be concise and helpful. Use bullet points where appropriate."

        user_prompt = (
            f"Question: {query}\n\n"
            f"Retrieved context:\n{context_string}\n\n"
            f"Provide a clear, grounded answer based on the above context and your learned lessons."
        )

        llm_answer = self.sarvam.chat(system_prompt, user_prompt)

        if llm_answer:
            summary = self._clean_answer(llm_answer)
        else:
            summary = self._fallback_summary(query, results, sources)

        # 5. Translate to Hindi if needed
        if detected_lang == "Hindi":
            translated = self.sarvam.translate(summary, "en-IN", "hi-IN")
            if translated:
                summary = translated

        # 6. Build confidence signal
        confidence = self._build_confidence(results)

        # 7. Voice (optional)
        voice = None
        if include_voice:
            tts_lang = "hi-IN" if detected_lang == "Hindi" else "en-IN"
            encoded = self.sarvam.text_to_speech(summary, tts_lang)
            if encoded:
                voice = {"audioBase64": encoded, "mode": "sarvam", "language": tts_lang}

        top = sources[0] if sources else None
        return {
            "query": query,
            "detectedLanguage": detected_lang,
            "summary": summary,
            "sources": sources,
            "topDocument": top["title"] if top else None,
            "topDocumentId": top["document_id"] if top else None,
            "confidence": confidence,
            "voicePlayback": voice,
            "appliedLessons": lessons
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_past_lessons(self, query_embedding: np.ndarray) -> list[str]:
        """Semantically retrieve past corrections or gold answers from the Experience Brain."""
        if not self.experience_store or self.experience_store.total_vectors == 0:
            return []

        lessons = []
        try:
            # Semantic search on experience brain
            # Threshold of 0.75 to ensure only highly relevant "past experiences" are used
            matches = self.experience_store.search(query_embedding, top_k=3)
            for m in matches:
                if m.get("score", 0) > 0.75:
                    correction = m.get("correction") or m.get("content")
                    if correction:
                        lessons.append(correction)
        except Exception as e:
            print(f"[rag_chain] Error retrieving semantic lessons: {e}")
            
        return list(set(lessons))[:3]

    def _build_sources(self, results: list[dict]) -> list[dict]:
        """De-duplicate results by document_id and build source list."""
        seen: set[str] = set()
        sources = []
        for r in results:
            doc_id = r["document_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            sources.append({
                "document_id": doc_id,
                "title": r["title"],
                "category": r.get("category", ""),
                "department": r.get("department", ""),
                "insurance_scheme": r.get("insurance_scheme", ""),
                "score": r["score"],
                "utility": r.get("utility_score", 0),
                "is_high_utility": r.get("utility_score", 0) > 5,
                "chunk_preview": r["chunk_text"][:200],
            })
        return sources

    def _augment_with_utility(self, results: list[dict]) -> list[dict]:
        """Fetch latest utility scores from SQLite for the retrieved documents."""
        if not results or not self.db_path or not self.db_path.exists():
            return results
        
        doc_ids = list(set(r["document_id"] for r in results))
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            placeholders = ",".join(["?"] * len(doc_ids))
            rows = cursor.execute(
                f"SELECT id, utility_score FROM documents WHERE id IN ({placeholders})",
                doc_ids
            ).fetchall()
            
            utility_map = {row["id"]: row["utility_score"] for row in rows}
            conn.close()
            
            for r in results:
                r["utility_score"] = utility_map.get(r["document_id"], 0)
        except Exception as e:
            print(f"[rag_chain] Error augmenting utility: {e}")
            
        return results

    def _format_context(self, results: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        for i, r in enumerate(results, 1):
            source_info = r['title']
            if r.get('source_file'):
                source_info += f" (File: {r['source_file']})"
            
            parts.append(
                f"[Source {i}: {source_info}]\n{r['chunk_text']}"
            )
        return "\n\n".join(parts)

    def _clean_answer(self, text: str) -> str:
        """Remove any model thinking tags and clean up the answer."""
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text or "No answer generated."

    def _fallback_summary(self, query: str, results: list[dict], sources: list[dict]) -> str:
        """Generate a template-based answer when Sarvam API is unavailable."""
        if not results:
            return "Sorry, I don't have this information in the uploaded hospital documents."
        top = results[0]
        doc_titles = ", ".join(s["title"] for s in sources[:3])
        return (
            f"Based on the hospital policy corpus, the most relevant documents are: {doc_titles}.\n\n"
            f"From '{top['title']}':\n{top['chunk_text'][:500]}"
        )

    def _detect_language(self, query: str, preferred_language: str) -> str:
        """Detect whether the query is in Hindi or English."""
        if preferred_language and preferred_language != "auto":
            return preferred_language.title()
        lowered = query.lower()
        hindi_tokens = ["kaunse", "chahiye", "samjhao", "ke liye", "kya", "in hindi", "hindi", "batao", "mein"]
        if any(token in lowered for token in hindi_tokens):
            return "Hindi"
        return "English"

    def _build_confidence(self, results: list[dict]) -> dict:
        """Build a confidence signal based on retrieval scores."""
        if not results:
            return {
                "level": "blocked",
                "label": "Out Of Corpus",
                "description": "No grounded answer: query not supported by the corpus.",
            }
        top_score = results[0]["score"]
        if top_score >= 0.7:
            return {
                "level": "exact",
                "label": "Strong Semantic Match",
                "description": "Answer is grounded in highly relevant document sections.",
            }
        if top_score >= 0.4:
            return {
                "level": "grounded",
                "label": "Good RAG Match",
                "description": "Answer is grounded in relevant retrieved content.",
            }
        return {
            "level": "semantic",
            "label": "Partial Match",
            "description": "Answer is based on loosely related content.",
        }

    def _no_info_response(self, query: str, detected_lang: str) -> dict:
        summary = "Sorry, I don't have this information in the uploaded hospital documents."
        if detected_lang == "Hindi":
            summary = "Sorry, mujhe yeh jankari uploaded hospital documents mein nahi mili."
        return {
            "query": query,
            "detectedLanguage": detected_lang,
            "summary": summary,
            "sources": [],
            "topDocument": None,
            "topDocumentId": None,
            "confidence": {
                "level": "blocked",
                "label": "Out Of Corpus",
                "description": "No grounded answer was returned.",
            },
            "voicePlayback": None,
        }
