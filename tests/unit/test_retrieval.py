from __future__ import annotations

import sqlite3

import pytest

from app.retrieval.search import _strip_filler, normalize_query, retrieve, search
from app.storage.schema import DDL, INDEXES, rebuild_fts
from app.utils.contracts import RetrievalStatus


def _seed_dataset(conn: sqlite3.Connection) -> None:
    conn.executemany(
        """
        INSERT INTO rooms (id, room_number, room_name, room_type, building_id, floor_id, lang, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """,
        [
            (1, "C105", "Robotics Lab", "Lab", "C", "1", "en"),
            (2, "C105", "معمل الروبوتات", "Lab", "C", "1", "ar"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO labs (id, lab_group_id, lab_name, building_id, floor_id, room_id, status, lang, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """,
        [
            (10, 2, "Robotics and Machine Vision", "C", "1", "C105", "Closed", "en"),
            (11, 2, "الروبوتات والرؤية", "C", "1", "C105", "Closed", "ar"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO departments (id, code, name, building, floor, room, description, lang, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """,
        [
            (20, "SET", "Software Engineering Department", "C", "1", "C111", "Software unit", "en"),
            (21, "SET", "قسم هندسة البرمجيات", "C", "1", "C111", "قسم البرمجيات", "ar"),
            (22, "SYE", "Systems Engineering Department", "C", "2", "C210", "Systems unit", "en"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO landmarks (id, landmark_name, building_id, floor_id, description, lang, is_active, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, datetime('now'))
        """,
        [
            (30, "Main Library", "Library", "1", "Quiet study area", "en"),
            (31, "المكتبة الرئيسية", "Library", "1", "مكان هادئ للمذاكرة", "ar"),
        ],
    )
    conn.execute(
        """
        INSERT INTO staff (id, source_staff_id, full_name, title, department_code, office_room, contact_notes, lang, is_active, updated_at)
        VALUES (40, '7', 'Dr. Sara Ali', 'Professor', 'SET', 'C112', 'Available after lectures', 'en', 1, datetime('now'))
        """
    )
    conn.executemany(
        """
        INSERT INTO office_hours (staff_full_name, weekday, start_time, end_time, notes, source_version, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        [
            ("Dr. Sara Ali", "Sunday", "10:00", "12:00", None, "OH-1"),
            ("Dr. Sara Ali", "Tuesday", "12:00", "14:00", None, "OH-2"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, lang)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("room", 1, "robot room", "robot room", "en"),
            ("room", 2, "معمل الروبوتات", "معمل الروبوتات", "ar"),
            ("landmark", 30, "library", "library", "en"),
            ("staff", 40, "dr sara", "dr sara", "en"),
            ("staff", 40, "د سارة", "د سارة", "ar"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO navigation_targets (target_type, canonical_id, nav_code, safety_notes, updated_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        """,
        [
            ("room", 1, "NAV_C105", "Use the east corridor"),
            ("department", 20, "NAV_SET", "Follow the admin hall"),
        ],
    )
    rebuild_fts(conn)


@pytest.fixture(autouse=True)
def retrieval_db(monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for statement in DDL + INDEXES:
        conn.execute(statement)
    _seed_dataset(conn)
    monkeypatch.setattr("app.retrieval.search.get_db", lambda: conn)
    return conn


def test_normalize_query_lowercases_english_only() -> None:
    assert normalize_query("Robotics Lab", "en") == "robotics lab"
    assert normalize_query("معمل الروبوتات", "ar") == "معمل الروبوتات"


def test_strip_filler_preserves_meaningful_terms() -> None:
    assert _strip_filler("where is the robotics lab", "en") == "robotics"
    assert _strip_filler("فين معمل الروبوتات", "ar") == "معمل الروبوتات"


def test_search_exact_room_number_in_english() -> None:
    result = search("c105", lang="en")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "room"
    assert result.canonical_name == "Robotics Lab"
    assert result.nav_code == "NAV_C105"


def test_search_arabic_alias_returns_arabic_room() -> None:
    result = search("معمل الروبوتات", lang="ar")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "room"
    assert result.canonical_name == "معمل الروبوتات"
    assert result.spoken_facts.room == "C105"


def test_search_staff_alias_preserves_staff_lookup_behavior() -> None:
    result = search("dr sara", lang="en", entity_type="staff")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "staff"
    assert result.canonical_name == "Dr. Sara Ali"
    assert "Sunday" in (result.spoken_facts.office_hours or "")


def test_search_staff_alias_in_arabic_uses_alias_table() -> None:
    result = search("د سارة", lang="ar", entity_type="staff")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "staff"
    assert result.canonical_name == "Dr. Sara Ali"


def test_search_fts_match_finds_english_lab() -> None:
    result = search("machine vision", lang="en", entity_type="lab")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "lab"
    assert result.canonical_name == "Robotics and Machine Vision"


def test_search_ambiguous_returns_candidates() -> None:
    result = search("engineering", lang="en", entity_type="department")

    assert result.status == RetrievalStatus.AMBIGUOUS
    assert "Software Engineering Department" in result.candidates
    assert "Systems Engineering Department" in result.candidates


def test_search_not_found_returns_not_found() -> None:
    result = search("quantum teleportation lab", lang="en")

    assert result.status == RetrievalStatus.NOT_FOUND


def test_retrieve_wrapper_remains_compatible() -> None:
    result = retrieve("library", lang="en")

    assert result.status == RetrievalStatus.OK
    assert result.entity_type == "landmark"
    assert result.canonical_name == "Main Library"
