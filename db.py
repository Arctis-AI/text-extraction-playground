import sqlite3

from config import DB_PATH


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS extractions (
                id TEXT PRIMARY KEY,
                batch_id TEXT,
                filename TEXT NOT NULL,
                file_size INTEGER,
                metadata TEXT,
                scan_detection TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS extraction_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id TEXT NOT NULL REFERENCES extractions(id),
                library TEXT NOT NULL,
                text TEXT,
                error TEXT,
                time_ms REAL,
                mode TEXT DEFAULT 'text'
            );
        """)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
