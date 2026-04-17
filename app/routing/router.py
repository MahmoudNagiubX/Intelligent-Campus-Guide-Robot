"""
Navigator - Intent Router
Classifies a final transcript into one of the four locked intent classes.

Flow:
1. Apply fast deterministic pre-rules (no LLM cost)
2. If pre-rules are inconclusive, call the Groq router
3. Parse and validate the JSON response
4. Return a typed IntentResult

Rules:
- Only final transcripts should be routed.
- A failed classification always returns IntentClass.UNKNOWN — never raises.
- The router never answers the user, only classifies.
"""

from __future__ import annotations

import atexit
import json
from pathlib import Path
from functools import lru_cache

from app.llm.groq_client import GroqClient
from app.utils.contracts import IntentClass, IntentResult
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt loader
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_router_prompt() -> str:
    path = Path("prompts/router_prompt.txt")
    if not path.exists():
        raise FileNotFoundError(f"Router prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Groq client singleton
# ─────────────────────────────────────────────────────────────────────────────

_groq: GroqClient | None = None

def _get_groq() -> GroqClient:
    global _groq
    if _groq is None:
        _groq = GroqClient()
    return _groq


def shutdown_router() -> None:
    """
    Close router-owned resources explicitly before interpreter teardown.
    """
    global _groq
    groq = _groq
    _groq = None

    if groq is None:
        return

    try:
        groq.close()
    except Exception:
        # Shutdown must never fall back into noisy destructor-time cleanup.
        pass


atexit.register(shutdown_router)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic pre-rules (checked before any LLM call)
# ─────────────────────────────────────────────────────────────────────────────

_NAV_TRIGGERS = [
    "take me to", "guide me to", "show me the way",
    "lead me to", "navigate to", "bring me to",
    "خدني", "وريني الطريق", "روحني",           # Egyptian Arabic navigation phrases
]

_CAMPUS_TRIGGERS = [
    "where is", "where are", "how do i find", "what floor",
    "office hours", "which building", "which room", "how to get to",
    "فين", "امتى", "ساعات العمل",              # Egyptian Arabic campus phrases
]

_SOCIAL_TRIGGERS = [
    "how are you", "what's up", "tell me a joke", "hello", "hi there",
    "good morning", "good afternoon", "what can you do", "who are you",
    "ازيك", "صباح الخير", "مساء الخير", "عامل ايه",  # Egyptian Arabic social
]


def _apply_pre_rules(text: str) -> IntentClass | None:
    """
    Fast keyword-based pre-classification.
    Returns an IntentClass if confident, or None to fall through to the LLM.
    """
    lower = text.lower()

    for trigger in _NAV_TRIGGERS:
        if trigger in lower:
            logger.debug("pre_rule_match", intent="navigation_request", trigger=trigger)
            return IntentClass.NAVIGATION_REQUEST

    for trigger in _CAMPUS_TRIGGERS:
        if trigger in lower:
            logger.debug("pre_rule_match", intent="campus_query", trigger=trigger)
            return IntentClass.CAMPUS_QUERY

    for trigger in _SOCIAL_TRIGGERS:
        if trigger in lower:
            logger.debug("pre_rule_match", intent="social_chat", trigger=trigger)
            return IntentClass.SOCIAL_CHAT

    return None


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing and validation
# ─────────────────────────────────────────────────────────────────────────────

_VALID_INTENTS = {c.value for c in IntentClass}
_VALID_LANGUAGES = {"en", "ar-EG", "mixed"}


def _parse_router_response(raw: str, original_text: str) -> IntentResult:
    """
    Parse and validate the JSON returned by the Groq router.
    Returns a safe IntentResult.UNKNOWN on any parsing failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("router_json_parse_error", error=str(exc), raw=raw[:200])
        return _unknown_result(original_text)

    intent_str = data.get("intent", "unknown")
    if intent_str not in _VALID_INTENTS:
        logger.warning("router_invalid_intent", received=intent_str)
        intent_str = "unknown"

    language = data.get("language", "en")
    if language not in _VALID_LANGUAGES:
        language = "en"

    target_text = data.get("target_text") or None
    if isinstance(target_text, str) and target_text.lower() in ("null", "none", ""):
        target_text = None

    clarification_q = data.get("clarification_question") or None
    if isinstance(clarification_q, str) and clarification_q.lower() in ("null", "none", ""):
        clarification_q = None

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return IntentResult(
        intent=IntentClass(intent_str),
        language=language,
        target_text=target_text,
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=clarification_q,
        confidence=confidence,
        raw_query=original_text,
        reason=data.get("reason"),
    )


def _unknown_result(text: str) -> IntentResult:
    return IntentResult(
        intent=IntentClass.UNKNOWN,
        language="en",
        raw_query=text,
        confidence=0.0,
        reason="fallback — classification failed",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def route(transcript: str) -> IntentResult:
    """
    Classify a final transcript into an IntentResult.

    Args:
        transcript: The final STT transcript text.

    Returns:
        IntentResult — always returns, never raises.
    """
    if not transcript or not transcript.strip():
        return _unknown_result(transcript)

    text = transcript.strip()
    logger.debug("router_start", text=text)

    # Step 1: fast pre-rules
    pre_intent = _apply_pre_rules(text)
    if pre_intent is not None:
        # Pre-rules don't extract target_text — let LLM fill that in
        # by still calling it, but we trust pre-rule for intent class
        pass  # fall through to LLM for target extraction

    # Step 2: LLM classification
    try:
        prompt = _load_router_prompt()
        groq = _get_groq()
        raw = groq.complete_json(system_prompt=prompt, user_message=text)
    except Exception as exc:
        logger.error("router_llm_error", error=str(exc))
        # If LLM fails but pre-rule fired, use pre-rule result without target
        if pre_intent is not None:
            return IntentResult(intent=pre_intent, raw_query=text,
                                confidence=0.7, reason="pre-rule fallback")
        return _unknown_result(text)

    if not raw:
        if pre_intent is not None:
            return IntentResult(intent=pre_intent, raw_query=text,
                                confidence=0.7, reason="pre-rule fallback")
        return _unknown_result(text)

    # Step 3: parse
    result = _parse_router_response(raw, text)

    # Step 4: if LLM disagrees with pre-rule, trust pre-rule for intent
    # but keep LLM's target_text and language
    if pre_intent is not None and result.intent != pre_intent:
        logger.debug(
            "router_pre_rule_override",
            pre_rule=pre_intent.value,
            llm=result.intent.value,
        )
        result = IntentResult(
            intent=pre_intent,
            language=result.language,
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