"""
Navigator - Database Connection Manager
Manages the SQLite connection for the entire application.

Rules:
- One connection, reused across the app lifetime.
- WAL mode enabled for safe concurrent reads.
- Foreign keys enforced.
- All access goes through get_db() — never open sqlite3 directly elsewhere.
"""

import sqlite3
from pathlib import Path

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_connection: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    """
    Return the active SQLite connection, creating it if needed.
    Called by every module that needs database access.
    """
    global _connection
    if _connection is None:
        _connection = _open_connection()
    return _connection


def _open_connection() -> sqlite3.Connection:
    """Open and configure the SQLite connection."""
    cfg = get_settings()
    db_path = Path(cfg.sqlite_db_path)

    # Ensure the directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("db_connecting", path=str(db_path))

    conn = sqlite3.connect(
        database=str(db_path),
        check_same_thread=False,   # Safe for our single-threaded pipeline
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )

    # Return rows as dict-like objects accessible by column name
    conn.row_factory = sqlite3.Row

    # Performance and integrity settings
    conn.execute("PRAGMA journal_mode=WAL;")    # Safe concurrent reads
    conn.execute("PRAGMA foreign_keys=ON;")     # Enforce FK constraints
    conn.execute("PRAGMA synchronous=NORMAL;")  # Balanced durability/speed
    conn.execute("PRAGMA temp_store=MEMORY;")   # Temp tables in RAM

    logger.info("db_connected", path=str(db_path))
    return conn


def close_db() -> None:
    """Close the database connection cleanly. Call during shutdown."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None
        logger.info("db_closed")
