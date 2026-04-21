import json

from app.config.settings import get_settings
from app.retrieval import ecu_knowledge
from app.retrieval.ecu_knowledge import search_ecu_knowledge


def _write_cache(tmp_path):
    path = tmp_path / "ecu_knowledge.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "title": "Faculty of Engineering and Technology",
                        "url": "https://ecu.edu.eg/faculties/engineering-and-technology/",
                        "content": "Engineering programs include software engineering.",
                        "keywords": ["engineering", "engineering faculty", "faculty of engineering"],
                    },
                    {
                        "title": "Faculty of Pharmacy and Drug Technology",
                        "url": "https://ecu.edu.eg/faculties/pharmacy-drug-technology/",
                        "content": "Pharmacy and pharmaceutical science programs.",
                        "keywords": ["pharmacy", "faculty of pharmacy"],
                    },
                    {
                        "title": "Library",
                        "url": "https://ecu.edu.eg/library/",
                        "content": "ECU Library provides academic resources.",
                        "keywords": ["library", "books"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_ecu_cache_matches_entries(monkeypatch, tmp_path) -> None:
    path = _write_cache(tmp_path)
    monkeypatch.setenv("ECU_KNOWLEDGE_PATH", str(path))
    get_settings.cache_clear()
    ecu_knowledge._load_ecu_knowledge.cache_clear()
    try:
        assert search_ecu_knowledge("engineering").title == "Faculty of Engineering and Technology"
        assert search_ecu_knowledge("pharmacy").title == "Faculty of Pharmacy and Drug Technology"
        assert search_ecu_knowledge("library").title == "Library"
        assert search_ecu_knowledge("engineering faculty").found is True
        assert search_ecu_knowledge("quantum teleportation").found is False
    finally:
        get_settings.cache_clear()
        ecu_knowledge._load_ecu_knowledge.cache_clear()
