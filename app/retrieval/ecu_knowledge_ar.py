"""
Arabic local ECU website knowledge cache.

Runtime searches data/ecu_knowledge_ar.json only. Refreshing the cache is done
by scripts/scrape_ecu_arabic.py outside live conversations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from app.pipeline.arabic_normalizer import normalize_arabic_transcript
from app.utils.logging import get_logger

logger = get_logger(__name__)

_KNOWLEDGE_PATH = Path("data/ecu_knowledge_ar.json")
_SIMILARITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class ECUKnowledgeArResult:
    found: bool
    content: str = ""
    title: str = ""
    source_url: str = ""
    confidence: float = 0.0


@lru_cache(maxsize=1)
def _load_ecu_knowledge_ar() -> dict:
    if not _KNOWLEDGE_PATH.exists():
        logger.warning("ecu_knowledge_ar.file_missing", path=str(_KNOWLEDGE_PATH))
        return {}
    try:
        data = json.loads(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("ecu_knowledge_ar.load_failed", error=str(exc))
        return {}
    logger.info("ecu_knowledge_ar.loaded", entries=len(data.get("entries", [])))
    return data


def search_ecu_knowledge_ar(query: str) -> ECUKnowledgeArResult:
    """Search the Arabic ECU knowledge cache."""
    if not query or len(query.strip()) < 2:
        return ECUKnowledgeArResult(found=False)

    entries = _load_ecu_knowledge_ar().get("entries", [])
    if not entries:
        return ECUKnowledgeArResult(found=False)

    normalized_query = normalize_arabic_transcript(query).lower()
    best_score = 0.0
    best_entry: dict | None = None

    for entry in entries:
        title = normalize_arabic_transcript(str(entry.get("title") or "")).lower()
        content = normalize_arabic_transcript(str(entry.get("content") or "")).lower()
        keywords = [normalize_arabic_transcript(str(keyword)).lower() for keyword in entry.get("keywords", [])]

        if normalized_query in keywords or any(normalized_query in keyword for keyword in keywords):
            return _result_from_entry(entry, 0.95)
        if normalized_query in content or normalized_query in title:
            score = 0.75
        else:
            title_score = SequenceMatcher(None, normalized_query, title).ratio()
            keyword_score = max(
                (SequenceMatcher(None, normalized_query, keyword).ratio() for keyword in keywords),
                default=0.0,
            )
            score = max(title_score, keyword_score)

        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= _SIMILARITY_THRESHOLD:
        return _result_from_entry(best_entry, best_score)
    return ECUKnowledgeArResult(found=False)


def _result_from_entry(entry: dict, confidence: float) -> ECUKnowledgeArResult:
    return ECUKnowledgeArResult(
        found=True,
        content=str(entry.get("content") or ""),
        title=str(entry.get("title") or ""),
        source_url=str(entry.get("url") or ""),
        confidence=confidence,
    )
