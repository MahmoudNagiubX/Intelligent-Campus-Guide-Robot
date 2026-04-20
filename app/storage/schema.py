"""
Navigator - SQLite Schema Bootstrap
All bilingual lookup tables include a lang column so retrieval can stay
language-scoped. FTS5 tables store lang as UNINDEXED to keep filtering fast
without polluting the text index.
"""

from __future__ import annotations

import sqlite3

from app.storage.db import get_db
from app.utils.logging import get_logger

logger = get_logger(__name__)


DDL: list[str] = [
    """CREATE TABLE IF NOT EXISTS departments (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        code          TEXT    NOT NULL,
        name          TEXT    NOT NULL,
        building      TEXT,
        floor         TEXT,
        room          TEXT,
        head_room     TEXT,
        description   TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT,
        UNIQUE(code, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS rooms (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        room_number   TEXT    NOT NULL,
        room_name     TEXT    NOT NULL,
        room_type     TEXT,
        building_id   TEXT    NOT NULL,
        floor_id      TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT,
        UNIQUE(building_id, room_number, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS labs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        lab_group_id  INTEGER,
        lab_name      TEXT    NOT NULL,
        building_id   TEXT    NOT NULL,
        floor_id      TEXT,
        room_id       TEXT,
        status        TEXT    DEFAULT 'Open',
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS floors (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id   TEXT    NOT NULL,
        floor_number  TEXT    NOT NULL,
        floor_name    TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT,
        UNIQUE(building_id, floor_number, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS landmarks (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        landmark_name TEXT    NOT NULL,
        building_id   TEXT,
        floor_id      TEXT,
        description   TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS staff (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        source_staff_id TEXT    UNIQUE,
        full_name       TEXT    NOT NULL UNIQUE,
        title           TEXT,
        department_code TEXT,
        office_room     TEXT,
        contact_notes   TEXT,
        lang            TEXT    NOT NULL DEFAULT 'en',
        is_active       INTEGER NOT NULL DEFAULT 1,
        updated_at      TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS office_hours (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        staff_full_name  TEXT NOT NULL,
        weekday          TEXT NOT NULL,
        start_time       TEXT NOT NULL,
        end_time         TEXT NOT NULL,
        notes            TEXT,
        source_version   TEXT,
        updated_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS aliases (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_type   TEXT NOT NULL,
        canonical_id     INTEGER NOT NULL,
        alias_text       TEXT NOT NULL,
        normalized_alias TEXT NOT NULL,
        lang             TEXT NOT NULL DEFAULT 'en',
        UNIQUE(canonical_type, canonical_id, normalized_alias, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS navigation_targets (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type   TEXT NOT NULL,
        canonical_id  INTEGER NOT NULL,
        nav_code      TEXT NOT NULL UNIQUE,
        safety_notes  TEXT,
        updated_at    TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sync_log (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name   TEXT NOT NULL,
        lang          TEXT,
        started_at    TEXT,
        finished_at   TEXT,
        rows_seen     INTEGER DEFAULT 0,
        rows_upserted INTEGER DEFAULT 0,
        rows_skipped  INTEGER DEFAULT 0,
        rows_errored  INTEGER DEFAULT 0,
        status        TEXT DEFAULT 'running',
        error_message TEXT
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_departments USING fts5(
        code, name, building, room,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_rooms USING fts5(
        room_number, room_name, room_type, building_id,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_labs USING fts5(
        lab_name, room_id, building_id,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_landmarks USING fts5(
        landmark_name, description, building_id,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_staff USING fts5(
        full_name, title, department_code, office_room,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
]

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_departments_lang ON departments(lang)",
    "CREATE INDEX IF NOT EXISTS idx_rooms_lang ON rooms(lang)",
    "CREATE INDEX IF NOT EXISTS idx_rooms_number ON rooms(room_number, lang)",
    "CREATE INDEX IF NOT EXISTS idx_labs_lang ON labs(lang)",
    "CREATE INDEX IF NOT EXISTS idx_labs_room ON labs(room_id, lang)",
    "CREATE INDEX IF NOT EXISTS idx_floors_lang ON floors(lang)",
    "CREATE INDEX IF NOT EXISTS idx_landmarks_lang ON landmarks(lang)",
    "CREATE INDEX IF NOT EXISTS idx_staff_lang ON staff(lang)",
    "CREATE INDEX IF NOT EXISTS idx_staff_source_id ON staff(source_staff_id)",
    "CREATE INDEX IF NOT EXISTS idx_office_hours_staff_name ON office_hours(staff_full_name)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_lang ON aliases(lang)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_normalized_lang ON aliases(normalized_alias, lang)",
    "CREATE INDEX IF NOT EXISTS idx_nav_targets_code ON navigation_targets(nav_code)",
]


def bootstrap_schema() -> None:
    """Create all tables, indexes, and FTS virtual tables. Idempotent."""
    conn = get_db()
    logger.info("schema_bootstrap_start", statement_count=len(DDL) + len(INDEXES))
    for statement in DDL + INDEXES:
        conn.execute(statement)
    conn.commit()
    logger.info("schema_bootstrap_complete")


def rebuild_fts(conn: sqlite3.Connection | None = None) -> None:
    """
    Rebuild all FTS5 indexes from the canonical tables.
    Called after bulk CSV sync so FTS stays aligned with the truth layer.
    """
    db = conn or get_db()

    db.execute("DELETE FROM fts_departments")
    db.execute(
        """
        INSERT INTO fts_departments(rowid, code, name, building, room, lang)
        SELECT id, code, COALESCE(name, ''), COALESCE(building, ''), COALESCE(room, ''), lang
        FROM departments
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_rooms")
    db.execute(
        """
        INSERT INTO fts_rooms(rowid, room_number, room_name, room_type, building_id, lang)
        SELECT id, room_number, room_name, COALESCE(room_type, ''), building_id, lang
        FROM rooms
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_labs")
    db.execute(
        """
        INSERT INTO fts_labs(rowid, lab_name, room_id, building_id, lang)
        SELECT id, lab_name, COALESCE(room_id, ''), building_id, lang
        FROM labs
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_landmarks")
    db.execute(
        """
        INSERT INTO fts_landmarks(rowid, landmark_name, description, building_id, lang)
        SELECT id, landmark_name, COALESCE(description, ''), COALESCE(building_id, ''), lang
        FROM landmarks
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_staff")
    db.execute(
        """
        INSERT INTO fts_staff(rowid, full_name, title, department_code, office_room, lang)
        SELECT id, full_name, COALESCE(title, ''), COALESCE(department_code, ''), COALESCE(office_room, ''), lang
        FROM staff
        WHERE is_active = 1
        """
    )

    db.commit()
    logger.info("fts_rebuilt")


def rebuild_fts_indexes() -> None:
    """Compatibility wrapper for legacy callers."""
    rebuild_fts()


def get_table_counts() -> dict[str, int]:
    """Return row counts for all core runtime tables."""
    conn = get_db()
    tables = [
        "departments",
        "rooms",
        "labs",
        "floors",
        "landmarks",
        "staff",
        "office_hours",
        "aliases",
        "navigation_targets",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    return counts
