"""
Arabic semantic query understanding for campus retrieval.

Pure string cleanup for Egyptian Arabic campus requests. It strips intent
phrases and returns the best entity text for Arabic retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRIP_PREFIXES: tuple[str, ...] = (
    r"(?:ممكن\s+تقولي|ممكن\s+تقولى|قولي|قولى)\s+(?:عن|فين|امتى|ازاي|ازاى)?\s*(?:هو|هي)?\s*",
    r"(?:عايز|عاوز|حابب|محتاج)\s+(?:اعرف|أعرف|اروح|أروح|اشوف|أشوف|ازور|أزور)\s*(?:عن|فين|ازاي|ازاى)?\s*",
    r"(?:ممكن\s+تساعدني|تساعدني)\s+(?:اوصل|أوصل|اروح|أروح|الوصول)\s*(?:لـ|ل|الى|إلى)?\s*",
    r"(?:خدني|وديني|روحني|وصلني|خديني|ودينى)\s*(?:لـ|ل|الى|إلى)?\s*",
    r"(?:فين|وين|امتى)\s*(?:هو|هي)?\s*(?:ال)?",
)
_STRIP_SUFFIXES: tuple[str, ...] = (
    r"\s+(?:فين|وين|امتى|ازاي|ازاى)\??$",
    r"\s+(?:هو|هي)\??$",
    r"\s+(?:لو سمحت|من فضلك)\??$",
)
_PREFIX_RE = re.compile(r"^\s*(?:" + "|".join(_STRIP_PREFIXES) + r")", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"(?:" + "|".join(_STRIP_SUFFIXES) + r")", re.IGNORECASE)
_PERSON_PREFIX_RE = re.compile(r"^(?:الدكتور|دكتور|دكتورة|الدكتورة|استاذ|أستاذ|استاذة|أستاذة)\s+", re.IGNORECASE)
_LEADING_TO_RE = re.compile(r"^(?:لـ|ل|الى|إلى)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ArabicUnderstoodQuery:
    """Output of the Arabic query understander."""

    raw_query: str
    entity_text: str
    router_entity: str
    best_entity: str
    query_type: str
    has_person_prefix: bool


def understand_arabic(
    raw_query: str,
    router_entity: str = "",
    router_confidence: float = 0.0,
    router_confidence_threshold: float = 0.75,
) -> ArabicUnderstoodQuery:
    """Extract the best Arabic entity string for retrieval."""
    cleaned = " ".join((raw_query or "").strip().split())
    stripped = _PREFIX_RE.sub("", cleaned).strip()
    stripped = _LEADING_TO_RE.sub("", stripped).strip()
    stripped = _SUFFIX_RE.sub("", stripped).strip(" .?!،؟")

    has_person_prefix = bool(_PERSON_PREFIX_RE.match(stripped) or _PERSON_PREFIX_RE.search(cleaned))
    if has_person_prefix:
        stripped = _PERSON_PREFIX_RE.sub("", stripped).strip()
        stripped = re.sub(r"\s+مكتبه\s*$", "", stripped).strip()

    query_type = _classify_query_type(cleaned, has_person_prefix)
    router_entity = (router_entity or "").strip()
    if router_entity and router_confidence >= router_confidence_threshold:
        best = router_entity
    elif stripped and len(stripped) >= 2:
        best = stripped
    else:
        best = cleaned

    return ArabicUnderstoodQuery(
        raw_query=raw_query,
        entity_text=stripped,
        router_entity=router_entity,
        best_entity=best,
        query_type=query_type,
        has_person_prefix=has_person_prefix,
    )


def _classify_query_type(text: str, has_person_prefix: bool) -> str:
    if has_person_prefix:
        return "person"
    if any(token in text for token in ("خدني", "وديني", "روحني", "وصلني", "اروح", "أروح")):
        return "navigation"
    if any(token in text for token in ("فين", "وين", "غرفة", "اوضة", "معمل", "مكتب", "قسم", "مبنى")):
        return "location"
    return "general"
