"""
Navigator - Phase 1 Step 1.6: Truth Layer Tests
Tests for CSV sync, retrieval engine, alias resolution, FTS5 search,
ambiguity handling, and not-found behavior.

Run with:
    pytest tests/unit/test_retrieval.py -v

These tests use an in-memory SQLite database so they never touch real data.
"""

import sqlite3
import pytest

from app.utils.contracts import RetrievalStatus
from app.retrieval.search import normalize_query, retrieve, _strip_filler


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """
    Patch get_db() to return a fresh in-memory SQLite database populated
    with a minimal but realistic campus dataset for every test.
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Enable foreign keys and create schema
    conn.executescript(_SCHEMA_SQL)
    conn.executescript(_SEED_SQL)
    conn.commit()

    monkeypatch.setattr("app.retrieval.search.get_db", lambda: conn)


# ─────────────────────────────────────────────────────────────────────────────
# In-memory schema (minimal, mirrors real schema.py)
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    building TEXT, floor TEXT, room TEXT, description TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    building TEXT, floor TEXT, room TEXT, description TEXT,
    department_id INTEGER REFERENCES departments(id),
    map_node TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS staff (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    title TEXT,
    department_id INTEGER REFERENCES departments(id),
    office_location_id INTEGER REFERENCES locations(id),
    contact_notes TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS office_hours (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    staff_id INTEGER NOT NULL REFERENCES staff(id),
    weekday TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    notes TEXT,
    source_version TEXT
);

CREATE TABLE IF NOT EXISTS facilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT, building TEXT, floor TEXT, room TEXT, description TEXT,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_type TEXT NOT NULL,
    canonical_id INTEGER NOT NULL,
    alias_text TEXT NOT NULL,
    normalized_alias TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_aliases_normalized ON aliases(normalized_alias);

CREATE TABLE IF NOT EXISTS navigation_targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL,
    canonical_id INTEGER NOT NULL,
    nav_code TEXT NOT NULL UNIQUE,
    safety_notes TEXT
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

_SEED_SQL = """
-- Departments
INSERT INTO departments (id, code, name, building, floor, room) VALUES
    (1, 'CS_DEPT',  'Computer Science Department',  'Building A', '3', '301'),
    (2, 'SE_DEPT',  'Software Engineering Department', 'Building A', '3', '305'),
    (3, 'AI_DEPT',  'Artificial Intelligence Department', 'Building A', '4', '401');

-- Locations
INSERT INTO locations (id, code, name, building, floor, room, description, map_node) VALUES
    (1, 'LAB_214', 'Robotics Lab', 'Building C', '2', '214', 'Main robotics lab', 'c2_r214'),
    (2, 'LAB_101', 'Computer Lab 101', 'Building A', '1', '101', 'General computing lab', 'a1_r101'),
    (3, 'LIB_MAIN', 'Main Library', 'Library Building', '1', NULL, 'University library', 'lib_main'),
    (4, 'CAFE_MAIN', 'Main Cafeteria', 'Student Center', '1', NULL, 'Campus cafeteria', 'cafe_main'),
    (5, 'SE_DEPT_OFF', 'Software Engineering Office', 'Building A', '3', '305', 'SE department', 'a3_r305'),
    (6, 'CS_DEPT_OFF', 'Computer Science Office', 'Building A', '3', '301', 'CS department', 'a3_r301');

-- Navigation targets
INSERT INTO navigation_targets (target_type, canonical_id, nav_code) VALUES
    ('location', 1, 'NAV_LAB_214'),
    ('location', 2, 'NAV_LAB_101'),
    ('location', 3, 'NAV_LIB_MAIN');

-- Staff
INSERT INTO staff (id, full_name, title, department_id, office_location_id, contact_notes) VALUES
    (1, 'Dr. Ahmed Samy',    'Assistant Professor', 2, 5, 'Available Sunday and Tuesday'),
    (2, 'Dr. Sara Ali',      'Associate Professor', 2, 5, 'Office hours posted on door'),
    (3, 'Dr. Mohamed Hassan','Professor',           1, 6, 'Available by appointment');

-- Office hours
INSERT INTO office_hours (staff_id, weekday, start_time, end_time) VALUES
    (1, 'Sunday',    '10:00', '12:00'),
    (1, 'Tuesday',   '14:00', '16:00'),
    (2, 'Monday',    '09:00', '11:00'),
    (3, 'Wednesday', '11:00', '13:00');

-- Facilities
INSERT INTO facilities (id, code, name, category, building, floor, description) VALUES
    (1, 'WIFI_MAIN', 'Main Campus Wi-Fi Zone', 'wifi', 'Library Building', '1', 'High-speed wireless'),
    (2, 'MEDICAL',   'Medical Center',         'health', 'Main Building', '2', 'Student health services');

-- Aliases
INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias) VALUES
    ('location', 1, 'robot room',       'robot room'),
    ('location', 1, 'robotics room',    'robotics room'),
    ('location', 1, 'lab 214',          'lab 214'),
    ('location', 1, 'room 214',         'room 214'),
    ('location', 3, 'library',          'library'),
    ('location', 3, 'the library',      'the library'),
    ('location', 4, 'cafeteria',        'cafeteria'),
    ('location', 4, 'canteen',          'canteen'),
    ('staff',    1, 'Dr Ahmed',         'dr ahmed'),
    ('staff',    1, 'Ahmed Samy',       'ahmed samy'),
    ('staff',    2, 'Dr Sara',          'dr sara'),
    ('facility', 1, 'wifi',             'wifi'),
    ('facility', 1, 'wi-fi',            'wi-fi'),
    ('facility', 1, 'internet',         'internet');

-- Rebuild FTS indexes
INSERT INTO fts_locations(fts_locations) VALUES('rebuild');
INSERT INTO fts_staff(fts_staff) VALUES('rebuild');
INSERT INTO fts_departments(fts_departments) VALUES('rebuild');
INSERT INTO fts_facilities(fts_facilities) VALUES('rebuild');
"""


# ─────────────────────────────────────────────────────────────────────────────
# Normalization tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeQuery:
    def test_lowercase(self):
        assert normalize_query("ROBOTICS LAB") == "robotics lab"

    def test_collapses_whitespace(self):
        assert normalize_query("  where   is  the  lab  ") == "where is the lab"

    def test_strips_punctuation(self):
        result = normalize_query("where is lab 214?!")
        assert "?" not in result
        assert "!" not in result

    def test_preserves_digits(self):
        assert "214" in normalize_query("room 214")

    def test_preserves_arabic(self):
        result = normalize_query("فين المعمل")
        assert "فين" in result
        assert "المعمل" in result

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_only_whitespace(self):
        assert normalize_query("   ") == ""


class TestStripFiller:
    def test_strips_common_filler(self):
        result = _strip_filler("where is the robotics lab")
        assert "where" not in result
        assert "is" not in result
        assert "the" not in result
        assert "robotics" in result
        assert "lab" not in result  # "lab" is a filler word in the set

    def test_preserves_content_words(self):
        result = _strip_filler("ahmed samy")
        assert "ahmed" in result
        assert "samy" in result

    def test_fallback_on_all_filler(self):
        # if all words are filler, should return original text
        result = _strip_filler("where is the")
        assert result == "where is the"


# ─────────────────────────────────────────────────────────────────────────────
# Alias lookup tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAliasLookup:
    def test_exact_alias_robot_room(self):
        result = retrieve("robot room")
        assert result.status == RetrievalStatus.OK
        assert "Robotics" in result.canonical_name
        assert result.confidence == 1.0
        assert result.matched_via == "alias"

    def test_exact_alias_lab_214(self):
        result = retrieve("lab 214")
        assert result.status == RetrievalStatus.OK
        assert result.nav_code == "NAV_LAB_214"

    def test_alias_room_214(self):
        result = retrieve("room 214")
        assert result.status == RetrievalStatus.OK
        assert "Robotics" in result.canonical_name

    def test_alias_library(self):
        result = retrieve("library")
        assert result.status == RetrievalStatus.OK
        assert "Library" in result.canonical_name

    def test_alias_cafeteria(self):
        result = retrieve("cafeteria")
        assert result.status == RetrievalStatus.OK
        assert "Cafeteria" in result.canonical_name

    def test_alias_canteen(self):
        result = retrieve("canteen")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type == "location"

    def test_alias_dr_ahmed(self):
        result = retrieve("Dr Ahmed")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type == "staff"
        assert "Ahmed" in result.canonical_name

    def test_alias_wifi(self):
        result = retrieve("wifi")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type == "facility"

    def test_alias_internet(self):
        result = retrieve("internet")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type == "facility"

    def test_alias_case_insensitive(self):
        # normalize_query lowercases, so alias lookup should match
        result = retrieve("LIBRARY")
        assert result.status == RetrievalStatus.OK


# ─────────────────────────────────────────────────────────────────────────────
# FTS5 search tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFTSSearch:
    def test_fts_robotics_lab(self):
        result = retrieve("Robotics Lab")
        assert result.status == RetrievalStatus.OK
        assert "Robotics" in result.canonical_name

    def test_fts_computer_lab(self):
        # FTS may match the Computer Science Department or Computer Lab 101 — both are valid.
        result = retrieve("Computer Lab")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type in ("location", "department")

    def test_fts_software_engineering(self):
        result = retrieve("Software Engineering department")
        assert result.status in (RetrievalStatus.OK, RetrievalStatus.AMBIGUOUS)
        # Both department and office may match — either is valid

    def test_fts_medical_center(self):
        result = retrieve("medical center")
        assert result.status == RetrievalStatus.OK
        assert result.entity_type == "facility"

    def test_fts_main_library(self):
        result = retrieve("main library")
        assert result.status == RetrievalStatus.OK


# ─────────────────────────────────────────────────────────────────────────────
# Navigation target tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNavigationTargets:
    def test_nav_code_returned_for_location(self):
        result = retrieve("Robotics Lab")
        assert result.status == RetrievalStatus.OK
        assert result.nav_code == "NAV_LAB_214"
        assert result.map_node == "c2_r214"

    def test_nav_code_returned_via_alias(self):
        result = retrieve("robot room")
        assert result.status == RetrievalStatus.OK
        assert result.nav_code == "NAV_LAB_214"

    def test_nav_code_library(self):
        result = retrieve("library")
        assert result.status == RetrievalStatus.OK
        assert result.nav_code == "NAV_LIB_MAIN"

    def test_cafeteria_no_nav_code(self):
        # Cafeteria has no nav target in seed data
        result = retrieve("cafeteria")
        assert result.status == RetrievalStatus.OK
        assert result.nav_code is None


# ─────────────────────────────────────────────────────────────────────────────
# Spoken facts hydration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSpokenFacts:
    def test_location_facts_populated(self):
        result = retrieve("Robotics Lab")
        assert result.status == RetrievalStatus.OK
        assert result.spoken_facts is not None
        assert result.spoken_facts.building == "Building C"
        assert result.spoken_facts.floor == "2"
        assert result.spoken_facts.room == "214"

    def test_staff_facts_include_office_hours(self):
        result = retrieve("Dr Ahmed")
        assert result.status == RetrievalStatus.OK
        assert result.spoken_facts is not None
        assert result.spoken_facts.office_hours is not None
        assert "Sunday" in result.spoken_facts.office_hours or "Tuesday" in result.spoken_facts.office_hours

    def test_facility_facts_populated(self):
        result = retrieve("medical center")
        assert result.status == RetrievalStatus.OK
        assert result.spoken_facts is not None
        assert result.spoken_facts.description is not None


# ─────────────────────────────────────────────────────────────────────────────
# Not-found behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestNotFound:
    def test_completely_unrelated_query(self):
        result = retrieve("quantum mechanics")
        assert result.status == RetrievalStatus.NOT_FOUND

    def test_empty_string(self):
        result = retrieve("")
        assert result.status == RetrievalStatus.NOT_FOUND

    def test_gibberish(self):
        result = retrieve("asdfghjklzxcvbnm")
        assert result.status == RetrievalStatus.NOT_FOUND

    def test_no_matching_entity_type(self):
        result = retrieve("zzznomatchzzz")
        assert result.status == RetrievalStatus.NOT_FOUND
