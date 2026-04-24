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


def test_navigation_paths_sync_populates_navigation_targets(csv_seeded_db) -> None:
    count = csv_seeded_db.execute("SELECT COUNT(*) FROM navigation_targets").fetchone()[0]
    assert count > 0


def test_department_and_room_search_return_nav_codes(csv_seeded_db) -> None:
    department = search("Software Engineering Department", lang="en")
    room = search("C105", lang="en")

    assert department.status == RetrievalStatus.OK
    assert department.nav_code is not None
    assert room.status == RetrievalStatus.OK
    assert room.nav_code is not None
