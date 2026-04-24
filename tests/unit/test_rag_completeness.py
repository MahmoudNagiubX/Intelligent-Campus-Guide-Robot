from __future__ import annotations

import pytest

from app.config.settings import get_settings
from app.retrieval.search import search
from app.storage.db import close_db, get_db
from app.storage.schema import bootstrap_schema
from app.storage.sync_csv import sync_all_csvs
from app.utils.contracts import RetrievalStatus


@pytest.fixture()
def csv_seeded_db(monkeypatch: pytest.MonkeyPatch, tmp_path):
    close_db()
    get_settings.cache_clear()
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "navigator.sqlite"))
    monkeypatch.setenv("CSV_ENGLISH_DIR", "data/csv_english")
    monkeypatch.setenv("CSV_ARABIC_DIR", "data/csv_arabic")
    get_settings.cache_clear()
    bootstrap_schema()
    sync_all_csvs()
    yield get_db()
    close_db()
    get_settings.cache_clear()


def _assert_ok(query: str, entity_type: str):
    result = search(query, lang="en")
    assert result.status == RetrievalStatus.OK
    assert result.entity_type == entity_type
    return result


def test_major_english_rag_categories_are_retrievable(csv_seeded_db) -> None:
    _assert_ok("Building C", "building")
    _assert_ok("C", "building")
    _assert_ok("Software Engineering Department", "department")
    _assert_ok("C105", "room")
    _assert_ok("105", "room")
    _assert_ok("Robotics Lab", "lab")

    staff = _assert_ok("Dr. Islam Mohamed", "staff")
    assert staff.spoken_facts is not None
    assert staff.spoken_facts.building is not None
    assert staff.spoken_facts.floor is not None
    assert staff.spoken_facts.office_hours is not None

    without_prefix = _assert_ok("Islam Mohamed", "staff")
    assert without_prefix.entity_id == staff.entity_id

    member = search("Innovtronics", lang="en")
    assert member.status == RetrievalStatus.OK
    assert member.entity_type == "member"


def test_nav_code_exists_for_synced_room_and_lab_navigation_targets(csv_seeded_db) -> None:
    conn = csv_seeded_db
    rows = conn.execute(
        """
        SELECT target_type, canonical_id
        FROM navigation_targets
        WHERE target_type IN ('room', 'lab')
        ORDER BY target_type, canonical_id
        """
    ).fetchall()
    assert rows

    for row in rows:
        if row["target_type"] == "room":
            value = conn.execute("SELECT room_number FROM rooms WHERE id=?", (row["canonical_id"],)).fetchone()[0]
        else:
            value = conn.execute("SELECT lab_name FROM labs WHERE id=?", (row["canonical_id"],)).fetchone()[0]
        result = search(value, lang="en", entity_type=row["target_type"])
        assert result.status == RetrievalStatus.OK
        assert result.nav_code is not None
