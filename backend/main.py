"""
FastAPI application entry point for PolicySarthi backend.

Run with:
    python -m backend.main
    # or
    uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload

The Streamlit frontend (frontend/main.py) connects to this server at
http://localhost:5000.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so relative imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Local imports
from backend.config import settings
from backend.database import initialize_database
from backend.vector_db.indexer import build_index
from backend.llm_support.sarvam_client import SarvamClient
from backend.llm_support.rag_chain import RAGChain
from backend.api.routes import router

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_PATH = BASE_DIR / "data" / "documents.json"
FAISS_INDEX_DIR = BASE_DIR / "data" / "faiss_index"

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="PolicySarthi — Hospital Policy & Ayushman Claim Assistant API",
    description="RAG pipeline: FAISS vector search + Sarvam LLM for multilingual hospital policy retrieval.",
    version="2.0.0",
)

# CORS — allow the Streamlit frontend and any local dev tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Initialise all services on server start."""

    # 1. SQLite database (users, documents, query_logs, feedback)
    print("[startup] Initialising SQLite database ...")
    db_state = initialize_database(BASE_DIR)

    # 2. FAISS vector index
    print("[startup] Building / loading FAISS index ...")
    faiss_store = build_index(
        documents_path=DOCUMENTS_PATH,
        index_dir=FAISS_INDEX_DIR,
        max_chunk_chars=settings.max_chunk_chars,
    )

    # 3. Sarvam AI client
    print("[startup] Initialising Sarvam AI client ...")
    sarvam = SarvamClient(
        api_key=settings.sarvam_api_key,
        chat_model=settings.chat_model,
    )

    # 4. RAG chain
    rag_chain = RAGChain(faiss_store=faiss_store, sarvam=sarvam)

    # 5. User lookup for auth
    user_lookup: dict[str, dict] = {}
    for user in db_state["users"]:
        user_lookup[user["username"]] = user

    # Store everything on app.state so routes can access it
    app.state.db_path = db_state["db_path"]
    app.state.storage_dir = db_state["storage_dir"]
    app.state.faiss_store = faiss_store
    app.state.sarvam = sarvam
    app.state.rag_chain = rag_chain
    app.state.user_lookup = user_lookup
    app.state.tokens = {}  # token → user dict

    print(f"[startup] Ready — {faiss_store.total_vectors} vectors indexed, Sarvam {'enabled' if sarvam.enabled else 'disabled'}")


# Mount the API router
app.include_router(router)


@app.get("/")
def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


# ── Run directly ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=5000, reload=True)
