"""
Navigator - Intent Router
Classifies a final transcript into one of the locked intent classes.
"""

from __future__ import annotations

import atexit
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.llm.groq_client import GroqClient
from app.utils.contracts import IntentClass, IntentResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_router_prompt() -> str:
    path = Path("prompts/router_prompt.txt")
    if not path.exists():
        raise FileNotFoundError(f"Router prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


_groq: GroqClient | None = None


def _get_groq() -> GroqClient:
    global _groq
    if _groq is None:
        _groq = GroqClient()
    return _groq


def shutdown_router() -> None:
    global _groq
    groq = _groq
    _groq = None
    if groq is None:
        return
    try:
        groq.close()
    except Exception:
        pass


atexit.register(shutdown_router)

_NAV_TRIGGERS = [
    "take me to",
    "guide me to",
    "show me the way",
    "lead me to",
    "navigate to",
    "bring me to",
    "خدني",
    "وديني",
    "روحني",
    "وريني الطريق",
]
_CAMPUS_TRIGGERS = [
    "where is",
    "where are",
    "how do i find",
    "what floor",
    "office hours",
    "which building",
    "which room",
    "how to get to",
    "فين",
    "امتى",
    "إمتى",
    "ساعات العمل",
]
_SOCIAL_TRIGGERS = [
    "how are you",
    "what's up",
    "tell me a joke",
    "hello",
    "hi there",
    "good morning",
    "good afternoon",
    "what can you do",
    "who are you",
    "ازيك",
    "عامل ايه",
    "عامل إيه",
    "صباح الخير",
    "مساء الخير",
]

_INTENT_MAP = {
    "campus_query": IntentClass.CAMPUS_QUERY,
    "navigation_request": IntentClass.NAVIGATION_REQUEST,
    "social_chat": IntentClass.SOCIAL_CHAT,
    "unknown": IntentClass.UNKNOWN,
    "campus query": IntentClass.CAMPUS_QUERY,
    "navigation request": IntentClass.NAVIGATION_REQUEST,
    "social chat": IntentClass.SOCIAL_CHAT,
    "campus_query      ": IntentClass.CAMPUS_QUERY,
    "campus_query ": IntentClass.CAMPUS_QUERY,
    "campusquery": IntentClass.CAMPUS_QUERY,
    "campus-query": IntentClass.CAMPUS_QUERY,
    "campus_query|navigation_request": IntentClass.UNKNOWN,
    "campus_query | navigation_request": IntentClass.UNKNOWN,
    "campus_query|social_chat": IntentClass.UNKNOWN,
    "campus_query|unknown": IntentClass.UNKNOWN,
    "campus_query|navigation": IntentClass.UNKNOWN,
    "campus_query|campus_query": IntentClass.CAMPUS_QUERY,
    "campus_query|social": IntentClass.UNKNOWN,
    "campus_query|chat": IntentClass.UNKNOWN,
    "campus_query|none": IntentClass.UNKNOWN,
    "campus_query|": IntentClass.UNKNOWN,
    "campus_query/campus_query": IntentClass.CAMPUS_QUERY,
    "campusquery ": IntentClass.CAMPUS_QUERY,
    "campus_query.": IntentClass.CAMPUS_QUERY,
    "navigationrequest": IntentClass.NAVIGATION_REQUEST,
    "socialchat": IntentClass.SOCIAL_CHAT,
    "campusquery": IntentClass.CAMPUS_QUERY,
    "campus_query\n": IntentClass.CAMPUS_QUERY,
    "navigation_request\n": IntentClass.NAVIGATION_REQUEST,
    "social_chat\n": IntentClass.SOCIAL_CHAT,
    "unknown\n": IntentClass.UNKNOWN,
    "campus_query\t": IntentClass.CAMPUS_QUERY,
    "navigation_request\t": IntentClass.NAVIGATION_REQUEST,
    "social_chat\t": IntentClass.SOCIAL_CHAT,
    "unknown\t": IntentClass.UNKNOWN,
    "campus_query | unknown": IntentClass.UNKNOWN,
    "campus_query or unknown": IntentClass.UNKNOWN,
    "campus_query or navigation_request": IntentClass.UNKNOWN,
    "campus_query or social_chat": IntentClass.UNKNOWN,
    "campus_query or unknown ": IntentClass.UNKNOWN,
    "campusqueryorunknown": IntentClass.UNKNOWN,
    "campus_query??": IntentClass.CAMPUS_QUERY,
    "campus_query!": IntentClass.CAMPUS_QUERY,
    "campus_query,": IntentClass.CAMPUS_QUERY,
    "campus_query;": IntentClass.CAMPUS_QUERY,
    "campus_query:": IntentClass.CAMPUS_QUERY,
    "campus query      ": IntentClass.CAMPUS_QUERY,
    "campus_query  ": IntentClass.CAMPUS_QUERY,
    "campus_query\r": IntentClass.CAMPUS_QUERY,
    "navigation_request\r": IntentClass.NAVIGATION_REQUEST,
    "social_chat\r": IntentClass.SOCIAL_CHAT,
    "unknown\r": IntentClass.UNKNOWN,
    "campus_query\r\n": IntentClass.CAMPUS_QUERY,
    "navigation_request\r\n": IntentClass.NAVIGATION_REQUEST,
    "social_chat\r\n": IntentClass.SOCIAL_CHAT,
    "unknown\r\n": IntentClass.UNKNOWN,
    "campus_query | none": IntentClass.UNKNOWN,
    "campus_query | null": IntentClass.UNKNOWN,
    "campus_query or null": IntentClass.UNKNOWN,
    "campus_query | nil": IntentClass.UNKNOWN,
    "campus query ": IntentClass.CAMPUS_QUERY,
    "navigation request ": IntentClass.NAVIGATION_REQUEST,
    "social chat ": IntentClass.SOCIAL_CHAT,
    "Campus_Query": IntentClass.CAMPUS_QUERY,
    "Navigation_Request": IntentClass.NAVIGATION_REQUEST,
    "Social_Chat": IntentClass.SOCIAL_CHAT,
    "Unknown": IntentClass.UNKNOWN,
}


def _normalize_language(value: object, fallback: str = "en") -> str:
    raw = str(value or fallback).strip()
    lowered = raw.lower()
    if lowered.startswith("ar"):
        return "ar"
    if lowered.startswith("en"):
        return "en"
    if lowered == "mixed":
        return fallback
    return fallback


def _apply_pre_rules(text: str) -> IntentClass | None:
    lowered = text.lower()
    for trigger in _NAV_TRIGGERS:
        if trigger in lowered:
            return IntentClass.NAVIGATION_REQUEST
    for trigger in _CAMPUS_TRIGGERS:
        if trigger in lowered:
            return IntentClass.CAMPUS_QUERY
    for trigger in _SOCIAL_TRIGGERS:
        if trigger in lowered:
            return IntentClass.SOCIAL_CHAT
    return None


def _parse_intent(value: object) -> IntentClass:
    raw = str(value or "unknown").strip()
    if raw in _INTENT_MAP:
        return _INTENT_MAP[raw]
    normalized = raw.lower().replace("_", " ").strip()
    if normalized == "campus query":
        return IntentClass.CAMPUS_QUERY
    if normalized == "navigation request":
        return IntentClass.NAVIGATION_REQUEST
    if normalized == "social chat":
        return IntentClass.SOCIAL_CHAT
    return IntentClass.UNKNOWN


def _parse_router_response(raw: str, original_text: str, fallback_language: str = "en") -> IntentResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("router_json_parse_error", error=str(exc), raw=raw[:200])
        return _unknown_result(original_text, fallback_language)

    intent = _parse_intent(data.get("intent"))
    language = _normalize_language(data.get("language"), fallback_language)
    target_text = data.get("target_text")
    if not target_text:
        target_text = data.get("target_entity")
    if isinstance(target_text, str) and target_text.lower() in {"null", "none", ""}:
        target_text = None

    clarification_question = data.get("clarification_question")
    if isinstance(clarification_question, str) and clarification_question.lower() in {"null", "none", ""}:
        clarification_question = None

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return IntentResult(
        intent=intent,
        language=language,
        target_text=target_text,
        needs_clarification=bool(data.get("needs_clarification", False) or data.get("clarification_needed", False)),
        clarification_question=clarification_question,
        confidence=confidence,
        raw_query=original_text,
        reason=data.get("reason"),
    )


def _unknown_result(text: str, language: str = "en") -> IntentResult:
    return IntentResult(
        intent=IntentClass.UNKNOWN,
        language=language,
        raw_query=text,
        confidence=0.0,
        reason="fallback - classification failed",
    )


def route(transcript: str, lang_hint: Optional[str] = None) -> IntentResult:
    """Classify a final transcript into an IntentResult."""
    if not transcript or not transcript.strip():
        return _unknown_result(transcript, _normalize_language(lang_hint))

    text = transcript.strip()
    fallback_language = _normalize_language(lang_hint)
    pre_intent = _apply_pre_rules(text)

    try:
        prompt = _load_router_prompt()
        groq = _get_groq()
        raw = groq.complete_json(system_prompt=prompt, user_message=text)
    except Exception as exc:
        logger.error("router_llm_error", error=str(exc))
        if pre_intent is not None:
            return IntentResult(intent=pre_intent, raw_query=text, language=fallback_language, confidence=0.7, reason="pre-rule fallback")
        return _unknown_result(text, fallback_language)

    if not raw:
        if pre_intent is not None:
            return IntentResult(intent=pre_intent, raw_query=text, language=fallback_language, confidence=0.7, reason="pre-rule fallback")
        return _unknown_result(text, fallback_language)

    result = _parse_router_response(raw, text, fallback_language)
    if pre_intent is not None and result.intent != pre_intent:
        result = IntentResult(
            intent=pre_intent,
            language=result.language or fallback_language,
            target_text=result.target_text,
            needs_clarification=result.needs_clarification,
            clarification_question=result.clarification_question,
            confidence=max(result.confidence, 0.75),
            raw_query=text,
            reason=f"pre-rule override: {result.reason}",
        )

    logger.info(
        "router_result",
        intent=result.intent.value,
        target=result.target_text,
        language=result.language,
        confidence=round(result.confidence, 2),
    )
    return result
