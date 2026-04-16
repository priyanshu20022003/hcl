"""
Document indexer — reads documents.json AND any PDFs in the data folder,
chunks content, embeds, and stores everything in a FAISS index.
"""

import json
from pathlib import Path

from .embeddings import get_embeddings
from .faiss_store import FaissStore


DEFAULT_CHUNK_SIZE = 500  # characters


def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split text into chunks respecting word boundaries."""
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        projected = current_len + len(word) + 1
        if projected > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = projected
    if current:
        chunks.append(" ".join(current))
    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"[indexer] WARNING: pypdf not installed, skipping {pdf_path.name}")
        return ""

    try:
        reader = PdfReader(str(pdf_path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if cleaned:
                pages.append(cleaned)
        full_text = "\n\n".join(pages)
        print(f"[indexer] Extracted {len(full_text)} chars from {pdf_path.name} ({len(reader.pages)} pages)")
        return full_text
    except Exception as exc:
        print(f"[indexer] ERROR extracting {pdf_path.name}: {exc}")
        return ""


def _index_json_documents(
    documents_path: Path,
    max_chunk_chars: int,
    all_texts: list[str],
    all_metadata: list[dict],
):
    """Read documents.json and add chunks to the index lists."""
    documents = json.loads(documents_path.read_text(encoding="utf-8"))

    for doc in documents:
        doc_id = doc["id"]
        title = doc["title"]
        summary = doc.get("summary", "")
        content = doc.get("content", "")

        # Title + summary as one chunk for broad matching
        header_text = f"{title}. {summary}"
        all_texts.append(header_text)
        all_metadata.append({
            "document_id": doc_id,
            "chunk_index": -1,
            "chunk_text": header_text,
            "title": title,
            "category": doc.get("category", ""),
            "department": doc.get("department", ""),
            "insurance_scheme": doc.get("insurance_scheme", ""),
            "source_type": "json",
        })

        # Content chunks
        chunks = chunk_text(content, max_chunk_chars)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadata.append({
                "document_id": doc_id,
                "chunk_index": idx,
                "chunk_text": chunk,
                "title": title,
                "category": doc.get("category", ""),
                "department": doc.get("department", ""),
                "insurance_scheme": doc.get("insurance_scheme", ""),
                "source_type": "json",
            })

    print(f"[indexer] Processed {len(documents)} JSON documents")


def _index_pdf_files(
    data_dir: Path,
    max_chunk_chars: int,
    all_texts: list[str],
    all_metadata: list[dict],
):
    """Find all PDFs in the data directory and add their chunks to the index."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("[indexer] No PDF files found in data directory")
        return

    print(f"[indexer] Found {len(pdf_files)} PDF file(s): {[p.name for p in pdf_files]}")

    for pdf_path in pdf_files:
        full_text = _extract_pdf_text(pdf_path)
        if not full_text.strip():
            print(f"[indexer] Skipping {pdf_path.name} — no text extracted")
            continue

        # Generate a document ID from the filename
        doc_id = f"PDF-{pdf_path.stem.upper().replace(' ', '_')[:30]}"
        title = pdf_path.stem.replace("_", " ").title()

        # First chunk: title / first 300 chars as summary
        summary_text = full_text[:300].strip()
        header_text = f"{title}. {summary_text}"
        all_texts.append(header_text)
        all_metadata.append({
            "document_id": doc_id,
            "chunk_index": -1,
            "chunk_text": header_text,
            "title": title,
            "category": "PDF Document",
            "department": "",
            "insurance_scheme": "",
            "source_type": "pdf",
            "source_file": pdf_path.name,
        })

        # Content chunks
        chunks = chunk_text(full_text, max_chunk_chars)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadata.append({
                "document_id": doc_id,
                "chunk_index": idx,
                "chunk_text": chunk,
                "title": title,
                "category": "PDF Document",
                "department": "",
                "insurance_scheme": "",
                "source_type": "pdf",
                "source_file": pdf_path.name,
            })

        print(f"[indexer] {pdf_path.name} -> {len(chunks)} chunks")


def build_index(
    documents_path: str | Path,
    index_dir: str | Path,
    max_chunk_chars: int = DEFAULT_CHUNK_SIZE,
    force_rebuild: bool = False,
) -> FaissStore:
    """Build (or load) a FAISS index from documents.json + PDF files.

    Args:
        documents_path: path to documents.json
        index_dir: directory to save / load the FAISS index
        max_chunk_chars: maximum characters per chunk
        force_rebuild: if True, rebuild even if index exists on disk

    Returns:
        A ready-to-query FaissStore instance.
    """
    index_dir = Path(index_dir)
    documents_path = Path(documents_path)
    data_dir = documents_path.parent  # backend/data/
    store = FaissStore()

    # 1. Discover potential files
    pdf_files = sorted(data_dir.glob("*.pdf"))
    current_pdf_names = {p.name for p in pdf_files}

    # 2. Check if we can skip rebuild
    if not force_rebuild and (index_dir / "index.faiss").exists():
        store.load(index_dir)
        
        # Smart Check: Are all current PDFs in the loaded index?
        indexed_files = {m.get("source_file") for m in store.metadata if m.get("source_type") == "pdf"}
        
        # Also check JSON file timestamp/existence
        if current_pdf_names == indexed_files:
            print(f"[indexer] Smart Discovery: No changes detected. Loaded {store.total_vectors} vectors.")
            return store
        else:
            diff = current_pdf_names - indexed_files
            print(f"[indexer] Smart Discovery: Found {len(diff)} new file(s) {list(diff)}. Rebuilding index...")
            store = FaissStore() # Reset for fresh index

    # 3. Build fresh index from all sources
    print("[indexer] Building FAISS index from documents.json + PDFs ...")

    all_texts: list[str] = []
    all_metadata: list[dict] = []

    # JSON documents
    _index_json_documents(documents_path, max_chunk_chars, all_texts, all_metadata)

    # PDF files
    _index_pdf_files(data_dir, max_chunk_chars, all_texts, all_metadata)

    # 4. Embed everything and build the index
    if all_texts:
        print(f"[indexer] Embedding {len(all_texts)} total chunks ...")
        embeddings = get_embeddings(all_texts)
        store.add(embeddings, all_metadata)
        store.save(index_dir)
        print(f"[indexer] FAISS index built with {store.total_vectors} vectors")
    else:
        print("[indexer] WARNING: No content found to index.")
        
    return store
