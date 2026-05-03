"""SQLite schema + helpers for runs.sqlite.

Single canonical store for both eval results and the memory layer
(per ARCHITECTURE.md §4). The sqlite-vec virtual table is created lazily
by ensure_vec_table() because the extension is only needed for the
memory-swap path.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "results" / "runs.sqlite"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path = DB_PATH) -> None:
    """Create all tables if missing. Idempotent."""
    schema_sql = SCHEMA_PATH.read_text()
    with connect(path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


def load_vec(conn: sqlite3.Connection) -> bool:
    """Load sqlite-vec extension. Returns True on success."""
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except sqlite3.OperationalError:
        return False


def ensure_vec_table(conn: sqlite3.Connection, dim: int = 768) -> bool:
    """Create vec_memory virtual table if sqlite-vec is loadable. Idempotent.

    Schema includes auxiliary columns (`+`-prefixed = stored but not vector-indexed)
    so a single virtual table can hold embeddings for both rolling summaries and
    structured memory_objects, distinguished by ref_type/ref_id.
    """
    if not load_vec(conn):
        return False
    conn.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory USING vec0(
          embedding float[{dim}],
          +ref_type TEXT,
          +ref_id TEXT
        )"""
    )
    conn.commit()
    return True


def insert_eval_run(conn: sqlite3.Connection, **fields) -> None:
    cols = list(fields.keys())
    placeholders = ", ".join(["?"] * len(cols))
    sql = f"INSERT INTO eval_runs ({', '.join(cols)}) VALUES ({placeholders})"
    conn.execute(sql, [fields[c] for c in cols])
    conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"initialized {DB_PATH}")
    with connect() as c:
        tables = [
            r["name"]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        print(f"tables: {tables}")
