import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass
class Settings:
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    chat_model: str = os.getenv("SARVAM_CHAT_MODEL", "sarvam-m")
    translate_target: str = os.getenv("SARVAM_DEFAULT_TARGET_LANG", "hi-IN")
    upload_dir: str = os.getenv("UPLOAD_DIR", "storage/uploads")
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "500"))
    # FAISS settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "data/faiss_index")


settings = Settings()
