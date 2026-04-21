"""
Local ECU website knowledge cache.

Runtime searches data/ecu_knowledge.json only. Refreshing the cache is handled
by scripts/scrape_ecu.py outside live conversations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_SIMILARITY_THRESHOLD = 0.55


@dataclass(frozen=True)
class ECUKnowledgeResult:
    """One match from the local ECU knowledge cache."""

    found: bool
    content: str = ""
    title: str = ""
    source_url: str = ""
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _load_ecu_knowledge() -> dict:
    """Load ECU knowledge JSON from disk."""
    path = Path(get_settings().ecu_knowledge_path)
    if not path.exists():
        logger.warning("ecu_knowledge.file_missing", path=str(path))
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("ecu_knowledge.load_failed", error=str(exc))
        return {}
    logger.info("ecu_knowledge.loaded", entries=len(data.get("entries", [])))
    return data


def search_ecu_knowledge(query: str) -> ECUKnowledgeResult:
    """Search the local ECU knowledge cache."""
    if not query or len(query.strip()) < 2:
        return ECUKnowledgeResult(found=False)

    entries = _load_ecu_knowledge().get("entries", [])
    if not entries:
        return ECUKnowledgeResult(found=False)

    q = " ".join(query.lower().split())
    best_score = 0.0
    best_entry: Optional[dict] = None

    for entry in entries:
        title = str(entry.get("title") or "").lower()
        keywords = [str(keyword).lower() for keyword in entry.get("keywords", [])]
        content = str(entry.get("content") or "").lower()

        if q in keywords or any(q in keyword for keyword in keywords):
            return _result_from_entry(entry, 0.95)
        if q in content:
            score = 0.70
        else:
            title_score = SequenceMatcher(None, q, title).ratio()
            keyword_score = max((SequenceMatcher(None, q, keyword).ratio() for keyword in keywords), default=0.0)
            score = max(title_score, keyword_score)

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= _SIMILARITY_THRESHOLD:
        return _result_from_entry(best_entry, best_score)
    return ECUKnowledgeResult(found=False)


def _result_from_entry(entry: dict, confidence: float) -> ECUKnowledgeResult:
    return ECUKnowledgeResult(
        found=True,
        content=str(entry.get("content") or ""),
        title=str(entry.get("title") or ""),
        source_url=str(entry.get("url") or ""),
        confidence=confidence,
    )
