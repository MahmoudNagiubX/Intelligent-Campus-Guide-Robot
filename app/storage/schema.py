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
        name_norm     TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT,
        UNIQUE(code, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS buildings (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id   TEXT    NOT NULL,
        building_name TEXT    NOT NULL,
        description   TEXT,
        lang          TEXT    NOT NULL DEFAULT 'en',
        is_active     INTEGER NOT NULL DEFAULT 1,
        updated_at    TEXT,
        UNIQUE(building_id, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS rooms (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        room_number   TEXT    NOT NULL,
        room_name     TEXT    NOT NULL,
        room_type     TEXT,
        building_id   TEXT    NOT NULL,
        floor_id      TEXT,
        room_name_norm TEXT,
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
        lab_name_norm TEXT,
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
        landmark_name_norm TEXT,
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
        full_name_norm   TEXT,
        lang            TEXT    NOT NULL DEFAULT 'en',
        is_active       INTEGER NOT NULL DEFAULT 1,
        updated_at      TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS office_hours (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        staff_id         TEXT,
        staff_full_name  TEXT NOT NULL,
        weekday          TEXT NOT NULL,
        start_time       TEXT NOT NULL,
        end_time         TEXT NOT NULL,
        notes            TEXT,
        source_version   TEXT,
        lang             TEXT NOT NULL DEFAULT 'en',
        updated_at       TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS aliases (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_type   TEXT NOT NULL,
        canonical_id     INTEGER NOT NULL,
        alias_text       TEXT NOT NULL,
        normalized_alias TEXT NOT NULL,
        alias_text_norm  TEXT,
        lang             TEXT NOT NULL DEFAULT 'en',
        UNIQUE(canonical_type, canonical_id, normalized_alias, lang)
    )""",
    """CREATE TABLE IF NOT EXISTS navigation_targets (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type   TEXT NOT NULL,
        canonical_id  INTEGER NOT NULL,
        nav_code      TEXT NOT NULL UNIQUE,
        safety_notes  TEXT,
        updated_at    TEXT,
        UNIQUE(target_type, canonical_id)
    )""",
    """CREATE TABLE IF NOT EXISTS members (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        member_id    TEXT,
        full_name    TEXT    NOT NULL,
        role         TEXT,
        team         TEXT    DEFAULT 'Innovtronics',
        bio          TEXT,
        lang         TEXT    NOT NULL DEFAULT 'en',
        is_active    INTEGER NOT NULL DEFAULT 1,
        updated_at   TEXT,
        UNIQUE(full_name, lang)
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
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_buildings USING fts5(
        building_id, building_name, description,
        lang UNINDEXED,
        tokenize='unicode61'
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
    """CREATE VIRTUAL TABLE IF NOT EXISTS fts_members USING fts5(
        full_name, role, team, bio,
        lang UNINDEXED,
        tokenize='unicode61'
    )""",
]

INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_buildings_lang ON buildings(lang)",
    "CREATE INDEX IF NOT EXISTS idx_buildings_id ON buildings(building_id, lang)",
    "CREATE INDEX IF NOT EXISTS idx_departments_lang ON departments(lang)",
    "CREATE INDEX IF NOT EXISTS idx_rooms_lang ON rooms(lang)",
    "CREATE INDEX IF NOT EXISTS idx_rooms_number ON rooms(room_number, lang)",
    "CREATE INDEX IF NOT EXISTS idx_labs_lang ON labs(lang)",
    "CREATE INDEX IF NOT EXISTS idx_labs_room ON labs(room_id, lang)",
    "CREATE INDEX IF NOT EXISTS idx_floors_lang ON floors(lang)",
    "CREATE INDEX IF NOT EXISTS idx_landmarks_lang ON landmarks(lang)",
    "CREATE INDEX IF NOT EXISTS idx_staff_lang ON staff(lang)",
    "CREATE INDEX IF NOT EXISTS idx_staff_source_id ON staff(source_staff_id)",
    "CREATE INDEX IF NOT EXISTS idx_office_hours_staff_id ON office_hours(staff_id)",
    "CREATE INDEX IF NOT EXISTS idx_office_hours_staff_name ON office_hours(staff_full_name)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_lang ON aliases(lang)",
    "CREATE INDEX IF NOT EXISTS idx_aliases_normalized_lang ON aliases(normalized_alias, lang)",
    "CREATE INDEX IF NOT EXISTS idx_nav_targets_code ON navigation_targets(nav_code)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_nav_targets_type_canonical ON navigation_targets(target_type, canonical_id)",
    "CREATE INDEX IF NOT EXISTS idx_members_lang ON members(lang)",
]

_NORMALIZED_COLS = [
    ("labs", "lab_name_norm", "TEXT"),
    ("rooms", "room_name_norm", "TEXT"),
    ("departments", "name_norm", "TEXT"),
    ("landmarks", "landmark_name_norm", "TEXT"),
    ("staff", "full_name_norm", "TEXT"),
    ("aliases", "alias_text_norm", "TEXT"),
]

_MIGRATION_COLS = [
    ("office_hours", "staff_id", "TEXT"),
    ("office_hours", "lang", "TEXT NOT NULL DEFAULT 'en'"),
]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column to a table only if it does not already exist."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        logger.info("schema_column_added", table=table, column=column)


def bootstrap_schema() -> None:
    """Create all tables, indexes, and FTS virtual tables. Idempotent."""
    conn = get_db()
    logger.info("schema_bootstrap_start", statement_count=len(DDL) + len(INDEXES))
    for statement in DDL:
        conn.execute(statement)
    for table, column, col_type in _NORMALIZED_COLS:
        _add_column_if_missing(conn, table, column, col_type)
    for table, column, col_type in _MIGRATION_COLS:
        _add_column_if_missing(conn, table, column, col_type)
    for statement in INDEXES:
        conn.execute(statement)
    conn.commit()
    logger.info("schema_bootstrap_complete")


def rebuild_fts(conn: sqlite3.Connection | None = None) -> None:
    """
    Rebuild all FTS5 indexes from the canonical tables.
    Called after bulk CSV sync so FTS stays aligned with the truth layer.
    """
    db = conn or get_db()
    _populate_normalized_arabic_columns(db)

    db.execute("DELETE FROM fts_buildings")
    db.execute(
        """
        INSERT INTO fts_buildings(rowid, building_id, building_name, description, lang)
        SELECT id, building_id, building_name, COALESCE(description, ''), lang
        FROM buildings
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_departments")
    db.execute(
        """
        INSERT INTO fts_departments(rowid, code, name, building, room, lang)
        SELECT id, code,
               CASE WHEN lang='ar' THEN COALESCE(name_norm, name, '') ELSE COALESCE(name, '') END,
               COALESCE(building, ''), COALESCE(room, ''), lang
        FROM departments
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_rooms")
    db.execute(
        """
        INSERT INTO fts_rooms(rowid, room_number, room_name, room_type, building_id, lang)
        SELECT id, room_number,
               CASE WHEN lang='ar' THEN COALESCE(room_name_norm, room_name, '') ELSE COALESCE(room_name, '') END,
               COALESCE(room_type, ''), building_id, lang
        FROM rooms
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_labs")
    db.execute(
        """
        INSERT INTO fts_labs(rowid, lab_name, room_id, building_id, lang)
        SELECT id,
               CASE WHEN lang='ar' THEN COALESCE(lab_name_norm, lab_name, '') ELSE COALESCE(lab_name, '') END,
               COALESCE(room_id, ''), building_id, lang
        FROM labs
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_landmarks")
    db.execute(
        """
        INSERT INTO fts_landmarks(rowid, landmark_name, description, building_id, lang)
        SELECT id,
               CASE WHEN lang='ar' THEN COALESCE(landmark_name_norm, landmark_name, '') ELSE COALESCE(landmark_name, '') END,
               COALESCE(description, ''), COALESCE(building_id, ''), lang
        FROM landmarks
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_staff")
    db.execute(
        """
        INSERT INTO fts_staff(rowid, full_name, title, department_code, office_room, lang)
        SELECT id,
               CASE WHEN lang='ar' THEN COALESCE(full_name_norm, full_name, '') ELSE COALESCE(full_name, '') END,
               COALESCE(title, ''), COALESCE(department_code, ''), COALESCE(office_room, ''), lang
        FROM staff
        WHERE is_active = 1
        """
    )

    db.execute("DELETE FROM fts_members")
    db.execute(
        """
        INSERT INTO fts_members(rowid, full_name, role, team, bio, lang)
        SELECT id, full_name, COALESCE(role, ''), COALESCE(team, 'Innovtronics'), COALESCE(bio, ''), lang
        FROM members
        WHERE is_active = 1
        """
    )

    db.commit()
    logger.info("fts_rebuilt")


def _populate_normalized_arabic_columns(db: sqlite3.Connection) -> None:
    """Populate normalized Arabic columns before rebuilding FTS."""
    from app.pipeline.arabic_normalizer import normalize_arabic_for_storage

    updates = [
        ("labs", "lab_name", "lab_name_norm"),
        ("rooms", "room_name", "room_name_norm"),
        ("departments", "name", "name_norm"),
        ("landmarks", "landmark_name", "landmark_name_norm"),
        ("staff", "full_name", "full_name_norm"),
        ("aliases", "alias_text", "alias_text_norm"),
    ]
    for table, source_col, target_col in updates:
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if target_col not in columns:
            continue
        for row in db.execute(f"SELECT id, {source_col} AS value FROM {table} WHERE lang='ar'"):
            db.execute(
                f"UPDATE {table} SET {target_col}=? WHERE id=?",
                (normalize_arabic_for_storage(row["value"] or ""), row["id"]),
            )


def rebuild_fts_indexes() -> None:
    """Compatibility wrapper for legacy callers."""
    rebuild_fts()


def get_table_counts() -> dict[str, int]:
    """Return row counts for all core runtime tables."""
    conn = get_db()
    tables = [
        "departments",
        "buildings",
        "rooms",
        "labs",
        "floors",
        "landmarks",
        "staff",
        "office_hours",
        "aliases",
        "navigation_targets",
        "members",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    return counts
