from __future__ import annotations

import pytest

from app.config.settings import get_settings
from app.retrieval.search import search
from app.storage.db import close_db
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
    yield
    close_db()
    get_settings.cache_clear()


def test_staff_hydration_includes_room_location_hours_and_title(csv_seeded_db) -> None:
    result = search("Dr. Islam Mohamed", lang="en", entity_type="staff")

    assert result.status == RetrievalStatus.OK
    assert result.spoken_facts is not None
    assert result.spoken_facts.building is not None
    assert result.spoken_facts.floor is not None
    assert result.spoken_facts.room == "C112"
    assert result.spoken_facts.office_hours is not None
    assert result.spoken_facts.title is not None
