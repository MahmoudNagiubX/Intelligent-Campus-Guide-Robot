"""
Navigator - Database Schema
Defines and creates all tables, FTS5 indexes, and triggers.

Call bootstrap_schema() once at startup to ensure all tables exist.
This is idempotent — safe to call on every boot.
"""

import sqlite3

from app.storage.db import get_db
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Tables DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_DEPARTMENTS = """
CREATE TABLE IF NOT EXISTS departments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    building        TEXT,
    floor           TEXT,
    room            TEXT,
    description     TEXT,
    head_name       TEXT,
    contact_email   TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_LOCATIONS = """
CREATE TABLE IF NOT EXISTS locations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    building        TEXT,
    floor           TEXT,
    room            TEXT,
    description     TEXT,
    department_id   INTEGER REFERENCES departments(id),
    map_node        TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_STAFF = """
CREATE TABLE IF NOT EXISTS staff (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name           TEXT    NOT NULL,
    title               TEXT,
    department_id       INTEGER REFERENCES departments(id),
    office_location_id  INTEGER REFERENCES locations(id),
    contact_notes       TEXT,
    is_active           INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_OFFICE_HOURS = """
CREATE TABLE IF NOT EXISTS office_hours (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id        INTEGER NOT NULL REFERENCES staff(id),
    weekday         TEXT    NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT    NOT NULL,
    notes           TEXT,
    source_version  TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_FACILITIES = """
CREATE TABLE IF NOT EXISTS facilities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    code            TEXT    NOT NULL UNIQUE,
    name            TEXT    NOT NULL,
    category        TEXT,
    building        TEXT,
    floor           TEXT,
    room            TEXT,
    description     TEXT,
    is_active       INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_ALIASES = """
CREATE TABLE IF NOT EXISTS aliases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_type  TEXT    NOT NULL,
    canonical_id    INTEGER NOT NULL,
    alias_text      TEXT    NOT NULL,
    normalized_alias TEXT   NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_aliases_normalized
    ON aliases(normalized_alias);
CREATE INDEX IF NOT EXISTS idx_aliases_canonical
    ON aliases(canonical_type, canonical_id);
"""

_DDL_NAVIGATION_TARGETS = """
CREATE TABLE IF NOT EXISTS navigation_targets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type     TEXT    NOT NULL,
    canonical_id    INTEGER NOT NULL,
    nav_code        TEXT    NOT NULL UNIQUE,
    safety_notes    TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_nav_targets_code
    ON navigation_targets(nav_code);
"""

_DDL_SYNC_LOG = """
CREATE TABLE IF NOT EXISTS sync_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name     TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    rows_seen       INTEGER DEFAULT 0,
    rows_upserted   INTEGER DEFAULT 0,
    rows_skipped    INTEGER DEFAULT 0,
    rows_errored    INTEGER DEFAULT 0,
    status          TEXT    NOT NULL DEFAULT 'running',
    error_message   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# FTS5 Virtual Tables
# ─────────────────────────────────────────────────────────────────────────────
# trigram tokenizer gives us substring and fuzzy matching for campus names.
# Each FTS table shadows a real table for fast full-text search.

_DDL_FTS_LOCATIONS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_locations
USING fts5(
    code,
    name,
    building,
    floor,
    room,
    description,
    content='locations',
    content_rowid='id',
    tokenize='trigram'
);
"""

_DDL_FTS_STAFF = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_staff
USING fts5(
    full_name,
    title,
    contact_notes,
    content='staff',
    content_rowid='id',
    tokenize='trigram'
);
"""

_DDL_FTS_DEPARTMENTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_departments
USING fts5(
    code,
    name,
    description,
    content='departments',
    content_rowid='id',
    tokenize='trigram'
);
"""

_DDL_FTS_FACILITIES = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_facilities
USING fts5(
    code,
    name,
    category,
    description,
    content='facilities',
    content_rowid='id',
    tokenize='trigram'
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Ordered list of all DDL blocks to execute
# ─────────────────────────────────────────────────────────────────────────────

_ALL_DDL: list[tuple[str, str]] = [
    ("departments",         _DDL_DEPARTMENTS),
    ("locations",           _DDL_LOCATIONS),
    ("staff",               _DDL_STAFF),
    ("office_hours",        _DDL_OFFICE_HOURS),
    ("facilities",          _DDL_FACILITIES),
    ("aliases",             _DDL_ALIASES),
    ("navigation_targets",  _DDL_NAVIGATION_TARGETS),
    ("sync_log",            _DDL_SYNC_LOG),
    ("fts_locations",       _DDL_FTS_LOCATIONS),
    ("fts_staff",           _DDL_FTS_STAFF),
    ("fts_departments",     _DDL_FTS_DEPARTMENTS),
    ("fts_facilities",      _DDL_FTS_FACILITIES),
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_schema() -> None:
    """
    Create all tables and FTS indexes if they don't already exist.
    Idempotent — safe to call on every application boot.
    """
    conn = get_db()
    logger.info("schema_bootstrap_start", table_count=len(_ALL_DDL))

    with conn:
        for name, ddl in _ALL_DDL:
            try:
                conn.executescript(ddl)
                logger.debug("schema_table_ok", table=name)
            except sqlite3.Error as exc:
                logger.error("schema_table_failed", table=name, error=str(exc))
                raise

    logger.info("schema_bootstrap_complete")


def rebuild_fts_indexes() -> None:
    """
    Rebuild all FTS5 content tables from their source tables.
    Call after bulk CSV sync to ensure FTS indexes are fresh.
    """
    conn = get_db()
    fts_tables = ["fts_locations", "fts_staff", "fts_departments", "fts_facilities"]

    logger.info("fts_rebuild_start")
    with conn:
        for table in fts_tables:
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild');")
            logger.debug("fts_rebuilt", table=table)

    logger.info("fts_rebuild_complete")


def get_table_counts() -> dict[str, int]:
    """
    Return row counts for all core tables.
    Useful for health checks and post-sync verification.
    """
    conn = get_db()
    tables = ["departments", "locations", "staff", "office_hours",
              "facilities", "aliases", "navigation_targets"]
    counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()
        counts[table] = row[0]
    return counts
