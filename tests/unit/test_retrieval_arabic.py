from __future__ import annotations

import sqlite3

import pytest

from app.retrieval.search import search
from app.storage.schema import DDL, INDEXES, rebuild_fts
from app.utils.contracts import RetrievalStatus


@pytest.fixture()
def arabic_retrieval_db(monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    for statement in DDL + INDEXES:
        conn.execute(statement)
    conn.execute(
        """
        INSERT INTO rooms (id, room_number, room_name, room_type, building_id, floor_id, lang, is_active, updated_at)
        VALUES (1, '214', 'Robotics Lab', 'Lab', 'C', '2', 'en', 1, datetime('now'))
        """
    )
    conn.execute(
        """
        INSERT INTO rooms (id, room_number, room_name, room_type, building_id, floor_id, lang, is_active, updated_at)
        VALUES (2, '214', 'معمل الروبوتات', 'معمل', 'C', '2', 'ar', 1, datetime('now'))
        """
    )
    conn.execute(
        """
        INSERT INTO labs (id, lab_group_id, lab_name, building_id, floor_id, room_id, status, lang, is_active, updated_at)
        VALUES (10, 1, 'Robotics Lab', 'C', '2', '214', 'Open', 'en', 1, datetime('now'))
        """
    )
    conn.execute(
        """
        INSERT INTO labs (id, lab_group_id, lab_name, building_id, floor_id, room_id, status, lang, is_active, updated_at)
        VALUES (11, 1, 'معمل الروبوتات', 'C', '2', '214', 'Open', 'ar', 1, datetime('now'))
        """
    )
    conn.execute(
        """
        INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, alias_text_norm, lang)
        VALUES ('lab', 10, 'معمل الروبوت', 'معمل الروبوت', 'معمل الروبوت', 'ar')
        """
    )
    conn.execute(
        """
        INSERT INTO navigation_targets (target_type, canonical_id, nav_code, updated_at)
        VALUES ('room', 1, 'NAV_ROOM_214', datetime('now'))
        """
    )
    rebuild_fts(conn)
    monkeypatch.setattr("app.retrieval.search.get_db", lambda: conn)
    return conn


def test_normalized_arabic_query_hits_normalized_entry(arabic_retrieval_db: sqlite3.Connection) -> None:
    result = search("فين معمل الروبوتات", lang="ar", entity_type="lab")

    assert result.status == RetrievalStatus.OK
    assert result.canonical_name == "معمل الروبوتات"


def test_cross_language_fallback_finds_english_entity(arabic_retrieval_db: sqlite3.Connection) -> None:
    result = search("the Robotics Lab", lang="ar", entity_type="lab")

    assert result.status == RetrievalStatus.OK
    assert result.canonical_name == "Robotics Lab"


def test_room_number_normalization_matches_same_entity(arabic_retrieval_db: sqlite3.Connection) -> None:
    arabic = search("اوضة 214", lang="ar", entity_type="room")
    english = search("room 214", lang="en", entity_type="room")

    assert arabic.status == RetrievalStatus.OK
    assert english.status == RetrievalStatus.OK
    assert arabic.spoken_facts.room == english.spoken_facts.room == "214"


def test_arabic_alias_resolves_canonical(arabic_retrieval_db: sqlite3.Connection) -> None:
    result = search("معمل الروبوت", lang="ar", entity_type="lab")

    assert result.status == RetrievalStatus.OK
    assert result.canonical_name == "Robotics Lab"
