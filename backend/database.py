import json
import sqlite3
from datetime import datetime
from pathlib import Path


SEED_USERS = [
    {
        "id": "USR-001",
        "username": "admin",
        "password": "admin123",
        "display_name": "Aditi Sharma",
        "role": "admin",
        "department": "Administration",
    },
    {
        "id": "USR-002",
        "username": "staff",
        "password": "staff123",
        "display_name": "Rahul Verma",
        "role": "staff",
        "department": "Insurance Desk",
    },
    {
        "id": "USR-003",
        "username": "auditor",
        "password": "audit123",
        "display_name": "Neha Iyer",
        "role": "auditor",
        "department": "Compliance",
    },
    {
        "id": "USR-004",
        "username": "user",
        "password": "user123",
        "display_name": "General User",
        "role": "user",
        "department": "General",
    },
]


def initialize_database(base_dir: Path):
    data_dir = base_dir / "data"
    storage_dir = base_dir / "storage" / "uploads"
    storage_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "app.db"

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()

    cursor.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            document_type TEXT NOT NULL,
            category TEXT NOT NULL,
            department TEXT NOT NULL,
            insurance_scheme TEXT NOT NULL,
            effective_date TEXT,
            language TEXT NOT NULL,
            version TEXT NOT NULL,
            summary TEXT NOT NULL,
            content TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            uploaded_by TEXT NOT NULL,
            access_roles TEXT NOT NULL DEFAULT 'admin,staff,auditor,user',
            utility_score INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        );

        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            detected_language TEXT NOT NULL,
            top_document TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_log_id INTEGER,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            top_document_id TEXT,
            top_document TEXT,
            rating INTEGER NOT NULL,
            comment TEXT,
            correction TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    cursor.executemany(
        """
        INSERT OR REPLACE INTO users (id, username, password, display_name, role, department)
        VALUES (:id, :username, :password, :display_name, :role, :department)
        """,
        SEED_USERS,
    )

    document_columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(documents)").fetchall()
    }
    if "access_roles" not in document_columns:
        cursor.execute(
            "ALTER TABLE documents ADD COLUMN access_roles TEXT NOT NULL DEFAULT 'admin,staff,auditor,user'"
        )
    if "utility_score" not in document_columns:
        cursor.execute(
            "ALTER TABLE documents ADD COLUMN utility_score INTEGER DEFAULT 0"
        )

    seed_documents = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))
    for document in seed_documents:
        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (
                id, title, document_type, category, department, insurance_scheme,
                effective_date, language, version, summary, content, file_name,
                file_path, last_updated, uploaded_by, access_roles, utility_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document["id"],
                document["title"],
                document.get("document_type", document["category"]),
                document["category"],
                document["department"],
                document.get("insurance_scheme", "General"),
                document.get("effective_date", document["last_updated"]),
                document.get("language", "en-IN"),
                document.get("version", "v1"),
                document["summary"],
                document["content"],
                f"{document['id']}.txt",
                str(storage_dir / f"{document['id']}.txt"),
                document["last_updated"],
                "USR-001",
                document.get("access_roles", "admin,staff,auditor,user"),
                0,
            ),
        )
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document["id"],))
        for index, chunk in enumerate(chunk_text(document["content"])):
            cursor.execute(
                "INSERT INTO chunks (document_id, chunk_index, content) VALUES (?, ?, ?)",
                (document["id"], index, chunk),
            )

    connection.commit()

    users = [
        dict(row)
        for row in cursor.execute(
            "SELECT id, username, password, display_name, role, department FROM users"
        ).fetchall()
    ]
    connection.close()

    return {
        "db_path": db_path,
        "storage_dir": storage_dir,
        "users": users,
    }


def chunk_text(text: str, max_chars: int = 500):
    words = text.split()
    chunks = []
    current = []
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


def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
