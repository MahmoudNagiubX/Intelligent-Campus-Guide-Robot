"""
English hybrid retrieval orchestrator.

Used for campus and navigation intents only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.pipeline.query_understander import UnderstoodQuery
from app.retrieval.ecu_knowledge import ECUKnowledgeResult, search_ecu_knowledge
from app.retrieval.search import normalize_query, search
from app.utils.contracts import RetrievalResult, RetrievalStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HybridResult:
    """Unified retrieval result for the English answer path."""

    answered_by: str
    db_result: Optional[RetrievalResult] = None
    ecu_result: Optional[ECUKnowledgeResult] = None
    understood: Optional[UnderstoodQuery] = None
    needs_llm_enhancement: bool = True
    latency_ms: float = 0.0


def retrieve_hybrid(understood: UnderstoodQuery, lang: str = "en", top_k: int = 3) -> HybridResult:
    """Run DB, ECU cache, then general fallback orchestration."""
    t0 = time.monotonic()
    db_result = _db_search_with_fallbacks(understood, lang, top_k)
    elapsed = lambda: (time.monotonic() - t0) * 1000

    if db_result.status == RetrievalStatus.OK:
        logger.info(
            "hybrid_retriever.db_hit",
            entity=understood.best_entity,
            canonical=db_result.canonical_name,
            confidence=round(db_result.confidence, 3),
            latency_ms=round(elapsed()),
        )
        return HybridResult("db", db_result=db_result, understood=understood, latency_ms=elapsed())

    if db_result.status == RetrievalStatus.AMBIGUOUS:
        logger.info("hybrid_retriever.db_ambiguous", entity=understood.best_entity, candidates=db_result.candidates)
        return HybridResult(
            "clarification",
            db_result=db_result,
            understood=understood,
            needs_llm_enhancement=False,
            latency_ms=elapsed(),
        )

    logger.info("hybrid_retriever.db_miss", entity=understood.best_entity, raw_query=understood.raw_query[:60])
    ecu_result = search_ecu_knowledge(understood.best_entity)
    if ecu_result.found:
        logger.info("hybrid_retriever.ecu_web_hit", entity=understood.best_entity, source=ecu_result.source_url)
        return HybridResult("ecu_web", ecu_result=ecu_result, understood=understood, latency_ms=elapsed())

    logger.info("hybrid_retriever.llm_general_fallback", entity=understood.best_entity, raw_query=understood.raw_query[:60])
    return HybridResult("llm_general", understood=understood, latency_ms=elapsed())


def _db_search_with_fallbacks(understood: UnderstoodQuery, lang: str, top_k: int) -> RetrievalResult:
    """Try understood entity variants against the DB in priority order."""
    candidates: list[str] = []
    for candidate in (understood.router_entity, understood.entity_text, understood.raw_query, understood.best_entity):
        candidate = (candidate or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_result: RetrievalResult | None = None
    for candidate in candidates:
        normalized = normalize_query(candidate, lang=lang)
        if len(normalized) < 2:
            continue
        result = search(normalized, lang=lang, top_k=top_k)
        last_result = result
        if result.status in (RetrievalStatus.OK, RetrievalStatus.AMBIGUOUS):
            return result

    return last_result or search(normalize_query(understood.best_entity, lang=lang), lang=lang, top_k=top_k)
