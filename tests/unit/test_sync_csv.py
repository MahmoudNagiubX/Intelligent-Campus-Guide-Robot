"""
Navigator - Phase 1 Step 1.6: CSV Sync Tests
Tests for the CSV ingestion engine covering all runtime CSV tables, validation,
normalization, FK resolution, idempotency, and error handling.

Run with:
    pytest tests/unit/test_sync_csv.py -v
"""

import csv
import io
import sqlite3
from pathlib import Path

import pytest

from app.storage.sync_csv import (
    normalize,
    normalize_for_search,
    str_or_none,
    _sync_departments,
    _sync_locations,
    _sync_staff,
    _sync_office_hours,
    _sync_facilities,
    _sync_aliases,
    _sync_navigation_targets,
    sync_all_csvs,
)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory DB fixture
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    building TEXT, floor TEXT, room TEXT, description TEXT,
    head_name TEXT, contact_email TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    building TEXT, floor TEXT, room TEXT, description TEXT,
    department_id INTEGER REFERENCES departments(id),
    map_node TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    title TEXT,
    department_id INTEGER REFERENCES departments(id),
    office_location_id INTEGER REFERENCES locations(id),
    contact_notes TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS office_hours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id INTEGER NOT NULL REFERENCES staff(id),
    weekday TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    notes TEXT, source_version TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS facilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT, building TEXT, floor TEXT, room TEXT, description TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_type TEXT NOT NULL,
    canonical_id INTEGER NOT NULL,
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_aliases_normalized ON aliases(normalized_alias);

CREATE TABLE IF NOT EXISTS navigation_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL,
    canonical_id INTEGER NOT NULL,
    nav_code TEXT NOT NULL UNIQUE,
    safety_notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    rows_seen INTEGER DEFAULT 0, rows_upserted INTEGER DEFAULT 0,
    rows_skipped INTEGER DEFAULT 0, rows_errored INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_locations USING fts5(
    code, name, building, floor, room, description,
    content='locations', content_rowid='id', tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_staff USING fts5(
    full_name, title, contact_notes,
    content='staff', content_rowid='id', tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_departments USING fts5(
    code, name, description,
    content='departments', content_rowid='id', tokenize='trigram'
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_facilities USING fts5(
    code, name, category, description,
    content='facilities', content_rowid='id', tokenize='trigram'
);
"""


@pytest.fixture
def db() -> sqlite3.Connection:
    """Fresh in-memory SQLite connection with the full Navigator schema."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizationHelpers:
    def test_normalize_strips_whitespace(self):
        assert normalize("  hello  world  ") == "hello world"

    def test_normalize_empty(self):
        assert normalize("") == ""

    def test_normalize_none(self):
        assert normalize(None) == ""

    def test_normalize_for_search_lowercases(self):
        assert normalize_for_search("Robotics Lab") == "robotics lab"

    def test_str_or_none_empty_returns_none(self):
        assert str_or_none("") is None
        assert str_or_none("   ") is None
        assert str_or_none(None) is None

    def test_str_or_none_value_returns_stripped(self):
        assert str_or_none("  hello  ") == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# Department sync
# ─────────────────────────────────────────────────────────────────────────────

class TestDepartmentSync:
    def test_inserts_department(self, db):
        rows = [{"code": "CS_DEPT", "name": "Computer Science", "building": "A"}]
        upserted, skipped, errored = _sync_departments(db, rows, "test")
        assert upserted == 1
        assert skipped == 0
        assert errored == 0
        row = db.execute("SELECT * FROM departments WHERE code='CS_DEPT';").fetchone()
        assert row is not None
        assert row["name"] == "Computer Science"

    def test_upsert_updates_existing(self, db):
        rows = [{"code": "CS_DEPT", "name": "Computer Science"}]
        _sync_departments(db, rows, "test")
        db.commit()

        rows2 = [{"code": "CS_DEPT", "name": "CS Department Updated"}]
        _sync_departments(db, rows2, "test")
        db.commit()

        row = db.execute("SELECT name FROM departments WHERE code='CS_DEPT';").fetchone()
        assert row["name"] == "CS Department Updated"

    def test_skips_missing_name(self, db):
        rows = [{"code": "BAD_DEPT", "name": ""}]
        upserted, skipped, errored = _sync_departments(db, rows, "test")
        assert skipped == 1
        assert upserted == 0

    def test_skips_missing_code(self, db):
        rows = [{"code": "", "name": "Something"}]
        upserted, skipped, errored = _sync_departments(db, rows, "test")
        assert skipped == 1

    def test_multiple_departments(self, db):
        rows = [
            {"code": "DEPT1", "name": "Dept One"},
            {"code": "DEPT2", "name": "Dept Two"},
            {"code": "", "name": "Bad row"},
        ]
        upserted, skipped, errored = _sync_departments(db, rows, "test")
        assert upserted == 2
        assert skipped == 1

    def test_idempotent_on_repeat_sync(self, db):
        rows = [{"code": "CS_DEPT", "name": "Computer Science"}]
        _sync_departments(db, rows, "test")
        db.commit()
        _sync_departments(db, rows, "test")
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM departments WHERE code='CS_DEPT';").fetchone()[0]
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Location sync
# ─────────────────────────────────────────────────────────────────────────────

class TestLocationSync:
    def _seed_dept(self, db):
        db.execute("INSERT INTO departments (code, name) VALUES ('CS_DEPT', 'CS');")
        db.commit()

    def test_inserts_location(self, db):
        rows = [{"code": "LAB_101", "name": "Computer Lab 101", "building": "A", "floor": "1", "room": "101"}]
        upserted, skipped, errored = _sync_locations(db, rows, "test")
        db.commit()
        assert upserted == 1
        row = db.execute("SELECT * FROM locations WHERE code='LAB_101';").fetchone()
        assert row["building"] == "A"
        assert row["floor"] == "1"

    def test_resolves_department_fk(self, db):
        self._seed_dept(db)
        rows = [{"code": "LAB_101", "name": "Lab 101", "department_code": "CS_DEPT"}]
        _sync_locations(db, rows, "test")
        db.commit()
        row = db.execute("SELECT department_id FROM locations WHERE code='LAB_101';").fetchone()
        assert row["department_id"] is not None

    def test_unknown_dept_code_sets_null_fk(self, db):
        rows = [{"code": "LAB_X", "name": "Lab X", "department_code": "NONEXISTENT"}]
        _sync_locations(db, rows, "test")
        db.commit()
        row = db.execute("SELECT department_id FROM locations WHERE code='LAB_X';").fetchone()
        assert row["department_id"] is None

    def test_skips_missing_code(self, db):
        rows = [{"code": "", "name": "Missing Code"}]
        upserted, skipped, errored = _sync_locations(db, rows, "test")
        assert skipped == 1


# ─────────────────────────────────────────────────────────────────────────────
# Staff sync
# ─────────────────────────────────────────────────────────────────────────────

class TestStaffSync:
    def _seed_location(self, db):
        db.execute("INSERT INTO locations (code, name) VALUES ('OFFICE_A', 'Office A');")
        db.commit()

    def test_inserts_staff(self, db):
        rows = [{"full_name": "Dr. Ahmed Samy", "title": "Professor"}]
        upserted, skipped, errored = _sync_staff(db, rows, "test")
        db.commit()
        assert upserted == 1
        row = db.execute("SELECT * FROM staff WHERE full_name='Dr. Ahmed Samy';").fetchone()
        assert row["title"] == "Professor"

    def test_updates_existing_staff(self, db):
        rows = [{"full_name": "Dr. Ahmed Samy", "title": "Lecturer"}]
        _sync_staff(db, rows, "test")
        db.commit()
        rows2 = [{"full_name": "Dr. Ahmed Samy", "title": "Senior Professor"}]
        _sync_staff(db, rows2, "test")
        db.commit()
        row = db.execute("SELECT title FROM staff WHERE full_name='Dr. Ahmed Samy';").fetchone()
        assert row["title"] == "Senior Professor"

    def test_resolves_office_location_fk(self, db):
        self._seed_location(db)
        rows = [{"full_name": "Dr. Test", "office_location_code": "OFFICE_A"}]
        _sync_staff(db, rows, "test")
        db.commit()
        row = db.execute("SELECT office_location_id FROM staff WHERE full_name='Dr. Test';").fetchone()
        assert row["office_location_id"] is not None

    def test_skips_missing_full_name(self, db):
        rows = [{"full_name": "", "title": "Professor"}]
        upserted, skipped, errored = _sync_staff(db, rows, "test")
        assert skipped == 1


# ─────────────────────────────────────────────────────────────────────────────
# Office hours sync
# ─────────────────────────────────────────────────────────────────────────────

class TestOfficeHoursSync:
    def _seed_staff(self, db) -> None:
        db.execute("INSERT INTO staff (id, full_name) VALUES (1, 'Dr. Ahmed Samy');")
        db.commit()

    def test_inserts_office_hours(self, db):
        self._seed_staff(db)
        rows = [{"staff_full_name": "Dr. Ahmed Samy", "weekday": "Sunday",
                 "start_time": "10:00", "end_time": "12:00"}]
        upserted, skipped, errored = _sync_office_hours(db, rows, "test")
        db.commit()
        assert upserted == 1
        row = db.execute("SELECT * FROM office_hours WHERE staff_id=1;").fetchone()
        assert row["weekday"] == "Sunday"

    def test_skips_unknown_staff(self, db):
        rows = [{"staff_full_name": "Unknown Person", "weekday": "Monday",
                 "start_time": "09:00", "end_time": "10:00"}]
        upserted, skipped, errored = _sync_office_hours(db, rows, "test")
        assert skipped == 1

    def test_replaces_existing_day_hours(self, db):
        self._seed_staff(db)
        rows = [{"staff_full_name": "Dr. Ahmed Samy", "weekday": "Sunday",
                 "start_time": "10:00", "end_time": "12:00"}]
        _sync_office_hours(db, rows, "test")
        db.commit()

        rows2 = [{"staff_full_name": "Dr. Ahmed Samy", "weekday": "Sunday",
                  "start_time": "14:00", "end_time": "16:00"}]
        _sync_office_hours(db, rows2, "test")
        db.commit()

        rows_found = db.execute(
            "SELECT * FROM office_hours WHERE staff_id=1 AND weekday='Sunday';"
        ).fetchall()
        assert len(rows_found) == 1
        assert rows_found[0]["start_time"] == "14:00"

    def test_skips_missing_required_fields(self, db):
        rows = [{"staff_full_name": "Dr. Ahmed Samy", "weekday": "",
                 "start_time": "10:00", "end_time": "12:00"}]
        upserted, skipped, errored = _sync_office_hours(db, rows, "test")
        assert skipped == 1


# ─────────────────────────────────────────────────────────────────────────────
# Facility sync
# ─────────────────────────────────────────────────────────────────────────────

class TestFacilitySync:
    def test_inserts_facility(self, db):
        rows = [{"code": "WIFI_MAIN", "name": "Main Wi-Fi Zone", "category": "wifi"}]
        upserted, skipped, errored = _sync_facilities(db, rows, "test")
        db.commit()
        assert upserted == 1
        row = db.execute("SELECT * FROM facilities WHERE code='WIFI_MAIN';").fetchone()
        assert row["category"] == "wifi"

    def test_upserts_existing_facility(self, db):
        rows = [{"code": "WIFI_MAIN", "name": "Wi-Fi Zone v1", "category": "wifi"}]
        _sync_facilities(db, rows, "test")
        db.commit()
        rows2 = [{"code": "WIFI_MAIN", "name": "Wi-Fi Zone v2", "category": "wifi"}]
        _sync_facilities(db, rows2, "test")
        db.commit()
        row = db.execute("SELECT name FROM facilities WHERE code='WIFI_MAIN';").fetchone()
        assert row["name"] == "Wi-Fi Zone v2"

    def test_skips_missing_required_fields(self, db):
        rows = [{"code": "", "name": "Nameless"}]
        upserted, skipped, errored = _sync_facilities(db, rows, "test")
        assert skipped == 1


# ─────────────────────────────────────────────────────────────────────────────
# Alias sync
# ─────────────────────────────────────────────────────────────────────────────

class TestAliasSync:
    def test_inserts_alias(self, db):
        rows = [{"canonical_type": "location", "canonical_id": "1", "alias_text": "robot room"}]
        upserted, skipped, errored = _sync_aliases(db, rows, "test")
        db.commit()
        assert upserted == 1
        row = db.execute("SELECT * FROM aliases WHERE normalized_alias='robot room';").fetchone()
        assert row is not None

    def test_deduplicates_on_repeat(self, db):
        rows = [{"canonical_type": "location", "canonical_id": "1", "alias_text": "robot room"}]
        _sync_aliases(db, rows, "test")
        db.commit()
        _sync_aliases(db, rows, "test")
        db.commit()
        count = db.execute("SELECT COUNT(*) FROM aliases WHERE normalized_alias='robot room';").fetchone()[0]
        assert count == 1

    def test_skips_missing_required_fields(self, db):
        rows = [{"canonical_type": "location", "canonical_id": "", "alias_text": "robot room"}]
        upserted, skipped, errored = _sync_aliases(db, rows, "test")
        assert skipped == 1

    def test_normalizes_alias_text(self, db):
        rows = [{"canonical_type": "location", "canonical_id": "1", "alias_text": "  Robot  Room  "}]
        _sync_aliases(db, rows, "test")
        db.commit()
        row = db.execute("SELECT * FROM aliases WHERE normalized_alias='robot room';").fetchone()
        assert row is not None
        assert row["alias_text"] == "Robot Room"


class TestNavigationTargetSync:
    def _seed_targets(self, db) -> None:
        db.execute("INSERT INTO locations (id, code, name) VALUES (1, 'LAB_214', 'Robotics Lab');")
        db.execute("INSERT INTO departments (id, code, name) VALUES (1, 'CS_DEPT', 'Computer Science Department');")
        db.execute("INSERT INTO facilities (id, code, name) VALUES (1, 'MEDICAL', 'Medical Center');")
        db.commit()

    def test_inserts_navigation_target(self, db):
        self._seed_targets(db)
        rows = [{"target_type": "location", "canonical_id": "1", "nav_code": "NAV_LAB_214"}]
        upserted, skipped, errored = _sync_navigation_targets(db, rows, "test")
        db.commit()

        assert upserted == 1
        assert skipped == 0
        assert errored == 0
        row = db.execute("SELECT * FROM navigation_targets WHERE nav_code='NAV_LAB_214';").fetchone()
        assert row is not None
        assert row["target_type"] == "location"

    def test_updates_existing_navigation_target(self, db):
        self._seed_targets(db)
        _sync_navigation_targets(
            db,
            [{"target_type": "facility", "canonical_id": "1", "nav_code": "NAV_MEDICAL", "safety_notes": "old"}],
            "test",
        )
        db.commit()

        _sync_navigation_targets(
            db,
            [{"target_type": "facility", "canonical_id": "1", "nav_code": "NAV_MEDICAL", "safety_notes": "new"}],
            "test",
        )
        db.commit()

        row = db.execute("SELECT safety_notes FROM navigation_targets WHERE nav_code='NAV_MEDICAL';").fetchone()
        assert row["safety_notes"] == "new"

    def test_skips_unknown_target_type(self, db):
        rows = [{"target_type": "staff", "canonical_id": "1", "nav_code": "NAV_BAD"}]
        upserted, skipped, errored = _sync_navigation_targets(db, rows, "test")
        assert upserted == 0
        assert skipped == 1
        assert errored == 0

    def test_skips_missing_canonical_record(self, db):
        rows = [{"target_type": "location", "canonical_id": "99", "nav_code": "NAV_MISSING"}]
        upserted, skipped, errored = _sync_navigation_targets(db, rows, "test")
        assert upserted == 0
        assert skipped == 1
        assert errored == 0


# ─────────────────────────────────────────────────────────────────────────────
# Full sync integration (sync_all_csvs)
# ─────────────────────────────────────────────────────────────────────────────

class TestSyncAllCsvs:
    def test_sync_all_with_real_csv_dir(self, monkeypatch, tmp_path):
        """
        Create a minimal CSV directory and verify sync_all_csvs runs cleanly.
        """
        # Write minimal CSVs
        (tmp_path / "departments.csv").write_text(
            "code,name,building\nTEST_DEPT,Test Department,Building X\n"
        )
        (tmp_path / "locations.csv").write_text(
            "code,name,building\nTEST_LOC,Test Location,Building X\n"
        )

        # Set up in-memory DB
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()

        monkeypatch.setattr("app.storage.sync_csv.get_db", lambda: conn)
        monkeypatch.setattr("app.storage.sync_csv.get_settings",
                            lambda: _FakeSettings(str(tmp_path)))
        monkeypatch.setattr("app.storage.schema.get_db", lambda: conn)

        results = sync_all_csvs()

        assert "departments" in results
        assert results["departments"]["upserted"] == 1
        assert results["locations"]["upserted"] == 1


class _FakeSettings:
    def __init__(self, csv_dir: str):
        self.csv_data_dir = csv_dir
