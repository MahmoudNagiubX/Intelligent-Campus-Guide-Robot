"""Test corrections and fuzzy matching for English campus retrieval."""

from app.storage.schema import bootstrap_schema
from app.storage.sync_csv import sync_all_csvs


def test_retrieval_corrections_match_misspelled_lab():
    from app.retrieval.search import search

    bootstrap_schema()
    sync_all_csvs()

    result = search("where is the robotic slab", lang="en")
    assert result.status.value == "ok"
    assert result.entity_type == "lab"
    assert "Robotics" in (result.canonical_name or "")


def test_fuzzy_matches_misspelled_department():
    from app.retrieval.search import search

    bootstrap_schema()
    sync_all_csvs()

    result = search("mectronics", lang="en")
    assert result.status.value == "ok"
    assert result.entity_type == "department"
    assert "Mechatronics" in (result.canonical_name or "")


def test_longest_correction_wins_for_building_see():
    from app.pipeline.controller import _apply_en_corrections
    from app.retrieval.search import normalize_query

    assert _apply_en_corrections("building see") == "building c"
    assert normalize_query("building see", "en") == "building c"
