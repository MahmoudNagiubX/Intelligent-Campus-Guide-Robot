"""
English semantic query understanding for campus retrieval.

This module extracts the campus entity from a natural English request before
FTS search. It does not call network services.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STRIP_PREFIXES: list[str] = [
    r"(?:can you )?(?:please )?(?:tell me (?:about|where|how to get to|the location of))",
    r"(?:i (?:want|need|would like) to (?:know about|find|locate|go to|visit|get to))",
    r"(?:i(?:'m| am) (?:looking for|trying to find|trying to get to))",
    r"(?:how do i (?:get to|find|reach|locate))",
    r"(?:what(?:'s| is) the (?:location of|address of|room (?:number )?(?:for|of)))",
    r"(?:where(?:'s| is)(?: the)?)",
    r"(?:show me(?: the)?)",
    r"(?:take me to(?: the)?)",
    r"(?:navigate to(?: the)?)",
    r"(?:find(?: the)?)",
    r"(?:i(?:'m| am) interested in(?: the)?)",
    r"(?:(?:can|could) you (?:show|take|guide|direct) me to(?: the)?)",
    r"(?:directions? to(?: the)?)",
    r"(?:(?:what|which) (?:building|floor|room|block) (?:is|has)(?: the)?)",
    r"(?:(?:do you know where|do you know)(?: the)?(?: is)?)",
    r"(?:i(?:'d| would) like to (?:visit|see|go to|know about)(?: the)?)",
    r"(?:can you show me(?: the)?)",
    r"(?:where does)",
]
_STRIP_SUFFIXES: list[str] = [
    r"(?:\s+(?:please|now|today)\.?$)",
    r"(?:\s+located\.?$)",
    r"(?:\s+(?:is|are)\?$)",
]
_PREFIX_RE = re.compile(r"^\s*(?:" + "|".join(_STRIP_PREFIXES) + r")\s*", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"(" + "|".join(_STRIP_SUFFIXES) + r")", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class UnderstoodQuery:
    """Output of the English query understander."""

    raw_query: str
    entity_text: str
    router_entity: str
    best_entity: str
    query_type: str
    has_person_prefix: bool


def understand(
    raw_query: str,
    router_entity: str = "",
    router_confidence: float = 0.0,
    router_confidence_threshold: float = 0.75,
) -> UnderstoodQuery:
    """Extract the best English entity string for retrieval."""
    cleaned = (raw_query or "").strip()
    stripped = _PREFIX_RE.sub("", cleaned).strip()
    stripped = _SUFFIX_RE.sub("", stripped).strip(" .?!,")
    stripped = _ARTICLE_RE.sub("", stripped).strip()

    has_person_prefix = bool(re.match(r"^(?:dr|prof|professor|doctor)\.?\s+", stripped, re.IGNORECASE))
    query_type = _classify_query_type(cleaned)

    router_entity = (router_entity or "").strip()
    if router_entity and router_confidence >= router_confidence_threshold:
        best = router_entity
    elif stripped and len(stripped) >= 2:
        best = stripped
    else:
        best = cleaned

    return UnderstoodQuery(
        raw_query=raw_query,
        entity_text=stripped,
        router_entity=router_entity,
        best_entity=best,
        query_type=query_type,
        has_person_prefix=has_person_prefix,
    )


def _classify_query_type(text: str) -> str:
    lowered = (text or "").lower()
    if any(token in lowered for token in ("dr ", "dr.", "prof ", "professor ", "doctor ")):
        return "person"
    if any(token in lowered for token in ("take me", "navigate", "guide me", "directions")):
        return "navigation"
    if any(token in lowered for token in ("where", "location", "room", "floor", "building", "lab")):
        return "location"
    return "general"
