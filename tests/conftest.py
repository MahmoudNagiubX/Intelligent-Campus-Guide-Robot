from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_test_db(tmp_path_factory: pytest.TempPathFactory):
    """Redirect default DB access to a temp file for the entire test session."""
    db_path = str(tmp_path_factory.mktemp("db") / "navigator_test.db")
    os.environ["SQLITE_DB_PATH"] = db_path
    os.environ["ENGLISH_ONLY_MODE"] = "false"

    from app.config.settings import get_settings
    from app.storage.db import close_db

    close_db()
    get_settings.cache_clear()
    yield
    close_db()
    get_settings.cache_clear()
