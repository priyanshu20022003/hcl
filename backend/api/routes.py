"""
FastAPI routes for the PolicySarthi RAG pipeline.

All endpoints maintain the same contract as the original Flask app so the
Streamlit frontend works without any changes.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile
from pydantic import BaseModel
import tempfile
import os
import sqlite3
from pathlib import Path
from datetime import datetime

from ..vector_db.embeddings import get_embedding
from ..vector_db.indexer import build_index
from ..config import settings

router = APIRouter(prefix="/api")


# ── Request / Response models ────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class QueryRequest(BaseModel):
    query: str
    language: str = "auto"
    include_voice: bool = False


class FeedbackRequest(BaseModel):
    query: str
    queryLogId: int | None = None
    topDocument: str | None = None
    topDocumentId: str | None = None
    rating: int
    comment: str | None = None
    correction: str | None = None


# ── Dependency: get app state from request ───────────────────────────
def get_state(request: Request):
    return request.app.state


def get_current_user(request: Request) -> dict:
    """Extract and validate the bearer token from the Authorization header."""
    state = request.app.state
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth_header.split(" ", 1)[1]
    user = state.tokens.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user


# ── Health ───────────────────────────────────────────────────────────
@router.get("/health")
def health_check(request: Request):
    state = get_state(request)
    return {
        "status": "ok",
        "service": "hospital-policy-assistant",
        "sarvamConfigured": state.sarvam.enabled,
        "vectorsIndexed": state.faiss_store.total_vectors,
    }


# ── Auth ─────────────────────────────────────────────────────────────
@router.post("/auth/login")
def login(body: LoginRequest, request: Request):
    import secrets
    state = get_state(request)

    username = body.username.strip()
    password = body.password.strip()
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    user = state.user_lookup.get(username)
    if not user or user["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(24)
    public_user = {
        "id": user["id"],
        "username": user["username"],
        "displayName": user["display_name"],
        "role": user["role"],
        "department": user["department"],
    }
    state.tokens[token] = public_user
    return {"token": token, "user": public_user}


@router.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}


# ── RAG Query (main endpoint) ───────────────────────────────────────
@router.post("/query")
def query_assistant(body: QueryRequest, request: Request, current_user: dict = Depends(get_current_user)):
    """Main RAG inference endpoint — FAISS retrieval + Sarvam LLM."""
    state = get_state(request)
    rag: object = state.rag_chain

    result = rag.answer(
        query=body.query,
        preferred_language=body.language,
        top_k=5,
        include_voice=body.include_voice,
    )

    # Log the query
    import sqlite3
    from datetime import datetime
    try:
        conn = sqlite3.connect(str(state.db_path))
        conn.execute(
            "INSERT INTO query_logs (user_id, query, detected_language, top_document, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                current_user["id"],
                body.query,
                result.get("detectedLanguage", "English"),
                result.get("topDocument", ""),
                datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return result


# ── Documents ────────────────────────────────────────────────────────
@router.get("/documents")
def list_documents(request: Request, current_user: dict = Depends(get_current_user)):
    import sqlite3
    state = get_state(request)
    conn = sqlite3.connect(str(state.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, title, document_type, category, department, insurance_scheme, "
        "effective_date, language, version, summary, last_updated, file_name "
        "FROM documents ORDER BY last_updated DESC"
    ).fetchall()
    conn.close()
    return {"documents": [dict(r) for r in rows]}


@router.get("/documents/{document_id}")
def get_document(document_id: str, request: Request, current_user: dict = Depends(get_current_user)):
    import sqlite3
    state = get_state(request)
    conn = sqlite3.connect(str(state.db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = [
        dict(c) for c in conn.execute(
            "SELECT chunk_index, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    ]
    conn.close()
    doc = dict(row)
    doc["chunks"] = chunks
    return doc


# ── Search ───────────────────────────────────────────────────────────
@router.get("/search")
def search_documents(request: Request, q: str = "", current_user: dict = Depends(get_current_user)):
    """Semantic search across the corpus using FAISS."""
    if not q.strip():
        return {"query": q, "results": []}

    state = get_state(request)
    from ..vector_db.embeddings import get_embedding
    query_embedding = get_embedding(q)
    results = state.faiss_store.search(query_embedding, top_k=10)

    # De-duplicate by document
    seen: set[str] = set()
    unique = []
    for r in results:
        if r["document_id"] not in seen:
            seen.add(r["document_id"])
            unique.append({
                "id": r["document_id"],
                "title": r["title"],
                "department": r.get("department", ""),
                "score": r["score"],
                "preview": r["chunk_text"][:300],
            })
    return {"query": q, "results": unique}


# ── Feedback ─────────────────────────────────────────────────────────
@router.post("/feedback")
def submit_feedback(body: FeedbackRequest, request: Request, current_user: dict = Depends(get_current_user)):
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query is required for feedback")

    state = get_state(request)
    conn = sqlite3.connect(str(state.db_path))
    conn.execute(
        "INSERT INTO feedback (query_log_id, user_id, query, top_document_id, top_document, rating, comment, correction, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            body.queryLogId,
            current_user["id"],
            body.query,
            body.topDocumentId,
            body.topDocument,
            body.rating,
            body.comment,
            body.correction,
            datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        ),
    )
    conn.commit()
    conn.close()

    # ── Real-time Learning (Update Experience Memory Brain & Utility) ────
    # If it's a correction or a high-quality (1) rating, learn from it
    if body.correction or body.rating == 1:
        try:
            # Update Document Utility if it was a high-quality match
            if body.rating == 1 and body.topDocumentId:
                conn = sqlite3.connect(str(state.db_path))
                conn.execute(
                    "UPDATE documents SET utility_score = utility_score + 1 WHERE id = ?",
                    (body.topDocumentId,)
                )
                conn.commit()
                conn.close()
                print(f"[learning] Document utility increased for: {body.topDocumentId}")

            # 1. Embed the query to use as a semantic key
            query_emb = get_embedding(body.query)
            
            # 2. Add as a semantic memory
            memory_data = {
                "query": body.query,
                "correction": body.correction or body.comment or "",
                "rating": body.rating,
                "type": "correction" if body.correction else "gold_answer",
                "content": body.correction if body.correction else "User-validated high-quality response."
            }
            state.experience_store.add_single(query_emb, memory_data)
            
            # 3. Persist to disk immediately
            state.experience_store.save(state.experience_dir)
            print(f"[learning] New experience memory saved for query: {body.query[:50]}...")
        except Exception as e:
            print(f"[learning] Error updating experience brain: {e}")

    return {"message": "Feedback saved and learned.", "appliedSignal": "positive" if body.rating > 0 else "negative"}


# ── Voice ────────────────────────────────────────────────────────────
@router.post("/voice/transcribe")
async def transcribe_voice(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    state = get_state(request)
    
    if not state.sarvam.enabled:
        raise HTTPException(status_code=503, detail="Sarvam API is not configured.")

    # Save the uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Call Sarvam STT (which expects a pathlib.Path)
        result = state.sarvam.speech_to_text(tmp_path)
        if not result or not result.get("transcript"):
            raise HTTPException(status_code=500, detail="Failed to transcribe audio.")
        return {"transcript": result["transcript"].strip(), "language": result.get("language_code")}
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@router.post("/ocr")
async def transcribe_ocr(request: Request, file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    state = get_state(request)
    
    if not state.sarvam.enabled:
        raise HTTPException(status_code=503, detail="Sarvam API is not configured.")

    # Save the uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Call Sarvam Document Intelligence
        text = state.sarvam.extract_document_text(tmp_path)
        if not text:
            raise HTTPException(status_code=500, detail="Failed to extract text from image.")
        return {"text": text.strip()}
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ── Dashboard ────────────────────────────────────────────────────────
@router.get("/dashboard")
def dashboard(request: Request, current_user: dict = Depends(get_current_user)):
    import sqlite3
    state = get_state(request)
    conn = sqlite3.connect(str(state.db_path))
    conn.row_factory = sqlite3.Row
    docs_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    query_count = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
    departments = [dict(r) for r in conn.execute(
        "SELECT department AS name, COUNT(*) AS count FROM documents GROUP BY department ORDER BY count DESC"
    ).fetchall()]
    top_queries = [dict(r) for r in conn.execute(
        "SELECT query AS label, COUNT(*) AS count FROM query_logs GROUP BY query ORDER BY count DESC LIMIT 5"
    ).fetchall()]
    conn.close()

    return {
        "stats": {
            "documentsIndexed": docs_count,
            "vectorsIndexed": state.faiss_store.total_vectors,
            "queriesHandled": query_count,
        },
        "departments": departments,
        "topQueries": top_queries,
        "roleView": current_user["role"],
    }


# ── Analytics ────────────────────────────────────────────────────────
@router.get("/analytics")
def analytics(request: Request, current_user: dict = Depends(get_current_user)):
    if current_user["role"] not in ("admin", "auditor"):
        raise HTTPException(status_code=403, detail="Forbidden for this role")

    import sqlite3
    state = get_state(request)
    conn = sqlite3.connect(str(state.db_path))
    conn.row_factory = sqlite3.Row
    top_queries = [dict(r) for r in conn.execute(
        "SELECT query, COUNT(*) AS count FROM query_logs GROUP BY query ORDER BY count DESC LIMIT 10"
    ).fetchall()]
    by_language = [dict(r) for r in conn.execute(
        "SELECT detected_language AS language, COUNT(*) AS count FROM query_logs GROUP BY detected_language ORDER BY count DESC"
    ).fetchall()]
    fb = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END),0) AS positive, "
        "COALESCE(SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END),0) AS negative, COUNT(*) AS total FROM feedback"
    ).fetchone()
    conn.close()
    return {
        "topQueries": top_queries,
        "languageBreakdown": by_language,
        "feedbackSummary": dict(fb) if fb else {"positive": 0, "negative": 0, "total": 0},
    }


# ── Sample queries ───────────────────────────────────────────────────
@router.get("/sample-queries")
def sample_queries(current_user: dict = Depends(get_current_user)):
    return {
        "queries": [
            "Patient ke Ayushman claim ke liye kaunse documents chahiye?",
            "Is pre-authorization required before planned admission?",
            "Why do Ayushman claims get rejected?",
            "Explain the Ayushman process in Hindi.",
            "What is the coverage under Ayushman Bharat?",
        ]
    }
# ── Admin ──────────────────────────────────────────────────────────
@router.post("/admin/reindex")
def reindex_knowledge_base(request: Request, current_user: dict = Depends(get_current_user)):
    """Manually trigger a full refresh of the FAISS index from documents.json + PDF data."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only admins can re-index the knowledge base")

    state = get_state(request)
    
    # 1. Update paths (must match main.py)
    BASE_DIR = Path(__file__).resolve().parent.parent
    DOCUMENTS_PATH = BASE_DIR / "data" / "documents.json"
    FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"

    print("[api] User-triggered re-indexing started...")
    try:
        # 2. Call build_index with force_rebuild=True
        new_store = build_index(
            documents_path=DOCUMENTS_PATH,
            index_dir=FAISS_INDEX_DIR,
            max_chunk_chars=settings.max_chunk_chars,
            force_rebuild=True,
            db_path=state.db_path
        )
        
        # 3. Hot-swap the store in the app state
        state.faiss_store = new_store
        state.rag_chain.faiss_store = new_store
        
        print(f"[api] Re-indexing complete. New vector count: {new_store.total_vectors}")
        return {
            "status": "success", 
            "message": "Knowledge base refreshed successfully.",
            "vectorsIndexed": new_store.total_vectors
        }
    except Exception as e:
        print(f"[api] ERROR during re-indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")
