"""
Arabic hybrid retrieval orchestrator.

Arabic uses a serial path: DB search, local Arabic ECU cache, then honest
general LLM fallback. It never starts the English preflight executor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.pipeline.arabic_query_understander import ArabicUnderstoodQuery
from app.retrieval.ecu_knowledge_ar import ECUKnowledgeArResult, search_ecu_knowledge_ar
from app.retrieval.search import normalize_query, search
from app.utils.contracts import RetrievalResult, RetrievalStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ArabicHybridResult:
    answered_by: str
    understood: ArabicUnderstoodQuery
    db_result: Optional[RetrievalResult] = None
    ecu_result: Optional[ECUKnowledgeArResult] = None
    needs_llm_enhancement: bool = True
    latency_ms: float = 0.0


def retrieve_arabic_hybrid(understood: ArabicUnderstoodQuery, top_k: int = 3) -> ArabicHybridResult:
    """Run Arabic DB, Arabic ECU cache, then general fallback orchestration."""
    t0 = time.monotonic()
    elapsed = lambda: (time.monotonic() - t0) * 1000
    db_result = _db_search_with_fallbacks(understood, top_k)

    if db_result.status == RetrievalStatus.OK:
        logger.info(
            "arabic_hybrid.db_hit",
            entity=understood.best_entity[:40],
            canonical=db_result.canonical_name,
            confidence=round(db_result.confidence, 3),
            latency_ms=round(elapsed()),
        )
        return ArabicHybridResult("db", understood=understood, db_result=db_result, latency_ms=elapsed())

    if db_result.status == RetrievalStatus.AMBIGUOUS:
        logger.info("arabic_hybrid.db_ambiguous", entity=understood.best_entity[:40], candidates=db_result.candidates)
        return ArabicHybridResult(
            "clarification",
            understood=understood,
            db_result=db_result,
            needs_llm_enhancement=False,
            latency_ms=elapsed(),
        )

    logger.info("arabic_hybrid.db_miss", entity=understood.best_entity[:40])
    ecu_result = search_ecu_knowledge_ar(understood.best_entity)
    if ecu_result.found:
        logger.info("arabic_hybrid.ecu_hit", entity=understood.best_entity[:40], source=ecu_result.source_url[:60])
        return ArabicHybridResult("ecu_web", understood=understood, ecu_result=ecu_result, latency_ms=elapsed())

    logger.info("arabic_hybrid.llm_general_fallback", entity=understood.best_entity[:40])
    return ArabicHybridResult("llm_general", understood=understood, latency_ms=elapsed())


def _db_search_with_fallbacks(understood: ArabicUnderstoodQuery, top_k: int) -> RetrievalResult:
    candidates: list[str] = []
    for candidate in (understood.router_entity, understood.entity_text, understood.best_entity, understood.raw_query):
        candidate = (candidate or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_result: RetrievalResult | None = None
    entity_type = "staff" if understood.has_person_prefix else "any"
    for candidate in candidates:
        normalized = normalize_query(candidate, lang="ar-EG")
        if len(normalized) < 2:
            continue
        result = search(normalized, lang="ar-EG", entity_type=entity_type, top_k=top_k)
        last_result = result
        if result.status in (RetrievalStatus.OK, RetrievalStatus.AMBIGUOUS):
            return result

    return last_result or search(normalize_query(understood.best_entity, lang="ar-EG"), lang="ar-EG", top_k=top_k)
