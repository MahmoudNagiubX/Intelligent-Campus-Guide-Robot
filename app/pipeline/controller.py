"""
Navigator - Dual-Path Conversation Controller

Receives final transcripts, detects language, classifies intent, dispatches to
retrieval or social handling, and returns a ResponsePacket for TTS/actions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Optional

from app.config import get_settings
from app.llm.groq_client import GroqClient
from app.pipeline.language_detector import detect_language, lang_is_arabic
from app.pipeline.response_composer import ResponseComposer
from app.retrieval.search import normalize_query, search
from app.routing.router import route
from app.utils.contracts import IntentClass, IntentResult, ResponsePacket, RetrievalResult, TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_TRANSCRIPT_CORRECTIONS = {
    "robotic slab": "robotics lab",
    "robot room": "robotics lab",
}
_CORRECTION_MIN_CONFIDENCE = 0.82
_CORRECTION_MIN_GAIN = 0.18
_WEAK_RETRIEVAL_THRESHOLD = 0.60
_WEAK_MARGIN_THRESHOLD = 0.12
_LOCATION_REQUEST_PREFIXES = (
    "where is",
    "where's",
    "take me",
    "go to",
    "navigate to",
    "find",
    "show me",
    "فين",
    "خدني",
    "وديني",
)
_INCOMPLETE_LOCATION_QUERIES = {
    "where",
    "where is",
    "where's",
    "take me",
    "take me to",
    "go",
    "go to",
    "find",
    "show me",
    "navigate",
    "navigate to",
    "فين",
    "خدني",
    "وديني",
}


class ConversationController:
    """Orchestrates one full conversation turn from transcript to response."""

    def __init__(self, groq: Optional[GroqClient] = None) -> None:
        self._groq = groq or GroqClient()
        self._composer = ResponseComposer(groq=self._groq)
        self._trace_hook: Optional[Callable[..., None]] = None

    def set_trace_hook(self, trace_hook: Optional[Callable[..., None]]) -> None:
        self._trace_hook = trace_hook

    def handle_transcript(self, event: TranscriptEvent) -> ResponsePacket:
        session_id = event.session_id
        text = (event.text or "").strip()
        cfg = get_settings()
        detected_lang = detect_language(
            text=text,
            deepgram_lang=event.language,
            deepgram_confidence=event.language_confidence,
            default_lang=cfg.default_language,
        )
        language = detected_lang.code

        logger.info(
            "controller_turn_start",
            text=text[:80],
            language=language,
            language_source=detected_lang.source,
            session_id=session_id,
        )

        if not text:
            packet = self._composer.compose_unknown_answer(language=language, session_id=session_id)
            self._trace_response(packet)
            return packet

        pre_router_packet = self._pre_router_quality_gate(text, language, session_id)
        if pre_router_packet is not None:
            self._trace_response(pre_router_packet)
            return pre_router_packet

        try:
            packet = self._dispatch(
                text=text,
                language=language,
                session_id=session_id,
                stt_confidence=event.confidence,
                detected_language=detected_lang.code,
            )
            self._trace_response(packet)
            return packet
        except Exception as exc:
            logger.error("controller_unhandled_error", error=str(exc), session_id=session_id)
            self._trace("error_occurred", session_id, source="controller", message=str(exc))
            packet = ResponsePacket(
                text="حصلت مشكلة عندي. حاول تسألني تاني." if lang_is_arabic(detected_lang) else "Something went wrong. Please try again.",
                language=language,
                session_id=session_id,
            )
            self._trace_response(packet)
            return packet

    def _dispatch(
        self,
        *,
        text: str,
        language: str,
        session_id: Optional[str],
        stt_confidence: float,
        detected_language: str,
    ) -> ResponsePacket:
        intent_result = self._classify(text, detected_language, session_id)

        logger.info(
            "controller_intent",
            intent=intent_result.intent.value,
            target=intent_result.target_text,
            confidence=round(intent_result.confidence, 2),
            needs_clarification=intent_result.needs_clarification,
            session_id=session_id,
        )
        self._trace(
            "intent_decided",
            session_id,
            intent=intent_result.intent.value,
            target=intent_result.target_text,
            confidence=round(intent_result.confidence, 2),
            needs_clarification=intent_result.needs_clarification,
        )

        if intent_result.needs_clarification and intent_result.clarification_question:
            return ResponsePacket(
                text=intent_result.clarification_question,
                language=language,
                session_id=session_id,
            )

        intent_result = self._normalize_intent_query(intent_result, stt_confidence, language, session_id)

        if intent_result.intent == IntentClass.CAMPUS_QUERY:
            return self._handle_campus_query(intent_result, text, language, session_id, stt_confidence)
        if intent_result.intent == IntentClass.NAVIGATION_REQUEST:
            return self._handle_navigation_request(intent_result, text, language, session_id, stt_confidence)
        if intent_result.intent == IntentClass.SOCIAL_CHAT:
            return self._handle_social_chat(text, language, session_id)
        return self._composer.compose_unknown_answer(language=language, session_id=session_id)

    def _handle_campus_query(
        self,
        intent_result: IntentResult,
        transcript_text: str,
        language: str,
        session_id: Optional[str],
        stt_confidence: float,
    ) -> ResponsePacket:
        query = intent_result.target_text or intent_result.raw_query or ""
        original_query = intent_result.raw_query or transcript_text or ""
        query, retrieval = self._resolve_retrieval_query(query, stt_confidence, language, session_id)
        self._trace(
            "retrieval_finished",
            session_id,
            path="campus",
            status=retrieval.status.value,
            entity=retrieval.canonical_name,
        )

        clarification_packet = self._maybe_build_quality_clarification(
            intent_result=intent_result,
            transcript_text=transcript_text,
            retrieval=retrieval,
            language=language,
            session_id=session_id,
        )
        if clarification_packet is not None:
            return clarification_packet

        return self._composer.compose_campus_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )

    def _handle_navigation_request(
        self,
        intent_result: IntentResult,
        transcript_text: str,
        language: str,
        session_id: Optional[str],
        stt_confidence: float,
    ) -> ResponsePacket:
        query = intent_result.target_text or intent_result.raw_query or ""
        original_query = intent_result.raw_query or transcript_text or ""
        query, retrieval = self._resolve_retrieval_query(query, stt_confidence, language, session_id)
        self._trace(
            "retrieval_finished",
            session_id,
            path="navigation",
            status=retrieval.status.value,
            entity=retrieval.canonical_name,
            nav_code=retrieval.nav_code,
        )

        clarification_packet = self._maybe_build_quality_clarification(
            intent_result=intent_result,
            transcript_text=transcript_text,
            retrieval=retrieval,
            language=language,
            session_id=session_id,
        )
        if clarification_packet is not None:
            return clarification_packet

        return self._composer.compose_navigation_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )

    def _handle_social_chat(self, text: str, language: str, session_id: Optional[str]) -> ResponsePacket:
        return self._composer.compose_social_answer(transcript=text, language=language, session_id=session_id)

    def _classify(self, text: str, language: str, session_id: Optional[str]) -> IntentResult:
        try:
            return route(text, lang_hint=language)
        except Exception as exc:
            logger.error("controller_router_error", error=str(exc), session_id=session_id)
            self._trace("error_occurred", session_id, source="router", message=str(exc))
            return IntentResult(
                intent=IntentClass.UNKNOWN,
                language=language,
                raw_query=text,
                reason="router_exception",
            )

    def _pre_router_quality_gate(
        self,
        text: str,
        language: str,
        session_id: Optional[str],
    ) -> Optional[ResponsePacket]:
        if not self._looks_malformed_location_query(text, language):
            return None
        logger.info("controller_transcript_low_quality", reason="malformed_location_query", text=text[:80])
        return self._composer.compose_quality_clarification(
            language=language,
            session_id=session_id,
            ask_location=True,
        )

    def _normalize_intent_query(
        self,
        intent_result: IntentResult,
        stt_confidence: float,
        language: str,
        session_id: Optional[str],
    ) -> IntentResult:
        if intent_result.intent not in (IntentClass.CAMPUS_QUERY, IntentClass.NAVIGATION_REQUEST):
            return intent_result
        query = intent_result.target_text or intent_result.raw_query or ""
        corrected_query, corrected = self._maybe_apply_transcript_correction(query, stt_confidence, language, session_id)
        if not corrected:
            return intent_result
        return replace(
            intent_result,
            target_text=corrected_query if intent_result.target_text else intent_result.target_text,
            raw_query=corrected_query if intent_result.target_text is None and intent_result.raw_query else intent_result.raw_query,
        )

    def _resolve_retrieval_query(
        self,
        query: str,
        stt_confidence: float,
        language: str,
        session_id: Optional[str],
    ) -> tuple[str, RetrievalResult]:
        corrected_query, corrected = self._maybe_apply_transcript_correction(query, stt_confidence, language, session_id)
        final_query = corrected_query if corrected else query
        return final_query, search(final_query, lang=language)

    def _maybe_apply_transcript_correction(
        self,
        query: str,
        stt_confidence: float,
        language: str,
        session_id: Optional[str],
    ) -> tuple[str, bool]:
        normalized_query = normalize_query(query, language)
        if not normalized_query:
            return query, False

        base_retrieval: Optional[RetrievalResult] = None
        if language.startswith("ar"):
            return query, False

        for confused_phrase, canonical_phrase in _TRANSCRIPT_CORRECTIONS.items():
            if confused_phrase not in normalized_query:
                continue
            if base_retrieval is None:
                base_retrieval = search(query, lang=language)
            if stt_confidence >= 0.92 and base_retrieval.status.value == "ok" and base_retrieval.confidence >= 0.9:
                continue

            candidate_query = normalized_query.replace(confused_phrase, canonical_phrase)
            candidate_retrieval = search(candidate_query, lang=language)
            if not self._should_accept_correction(base_retrieval, candidate_retrieval):
                continue

            logger.info(
                "controller_transcript_normalized",
                before=query,
                after=candidate_query,
                canonical_name=candidate_retrieval.canonical_name,
                base_confidence=round(base_retrieval.confidence, 3),
                corrected_confidence=round(candidate_retrieval.confidence, 3),
                session_id=session_id,
            )
            return candidate_query, True
        return query, False

    @staticmethod
    def _should_accept_correction(base_retrieval: RetrievalResult, candidate_retrieval: RetrievalResult) -> bool:
        if candidate_retrieval.status.value != "ok":
            return False
        if candidate_retrieval.confidence < _CORRECTION_MIN_CONFIDENCE:
            return False
        if base_retrieval.status.value != "ok":
            return True
        return candidate_retrieval.confidence >= (base_retrieval.confidence + _CORRECTION_MIN_GAIN)

    def _maybe_build_quality_clarification(
        self,
        *,
        intent_result: IntentResult,
        transcript_text: str,
        retrieval: RetrievalResult,
        language: str,
        session_id: Optional[str],
    ) -> Optional[ResponsePacket]:
        if retrieval.status.value == "ambiguous":
            return None

        ask_location = intent_result.intent == IntentClass.NAVIGATION_REQUEST or self._looks_like_location_request(
            transcript_text, language
        )
        suggestion = retrieval.candidates[0] if retrieval.candidates else None
        alternatives = retrieval.candidates[:2]

        if retrieval.status.value == "not_found":
            if suggestion and retrieval.confidence >= 0.48:
                return self._composer.compose_quality_clarification(
                    language=language,
                    session_id=session_id,
                    suggestion=suggestion,
                    ask_location=ask_location,
                )
            if ask_location and self._has_weak_campus_evidence(intent_result.raw_query or transcript_text, language):
                return self._composer.compose_quality_clarification(
                    language=language,
                    session_id=session_id,
                    ask_location=True,
                )
            return None

        if retrieval.status.value == "ok":
            if ask_location and retrieval.confidence < _WEAK_RETRIEVAL_THRESHOLD and suggestion is not None:
                return self._composer.compose_quality_clarification(
                    language=language,
                    session_id=session_id,
                    suggestion=suggestion,
                    ask_location=True,
                )
            if (
                ask_location
                and len(alternatives) >= 2
                and retrieval.second_best_score >= _WEAK_RETRIEVAL_THRESHOLD
                and retrieval.score_margin <= _WEAK_MARGIN_THRESHOLD
            ):
                return self._composer.compose_quality_clarification(
                    language=language,
                    session_id=session_id,
                    alternatives=alternatives,
                    ask_location=True,
                )
        return None

    @staticmethod
    def _looks_like_location_request(text: str, language: str) -> bool:
        normalized = normalize_query(text, language)
        return normalized.startswith(_LOCATION_REQUEST_PREFIXES) or "location" in normalized

    @staticmethod
    def _looks_malformed_location_query(text: str, language: str) -> bool:
        normalized = normalize_query(text, language)
        if not normalized:
            return False
        if normalized in _INCOMPLETE_LOCATION_QUERIES:
            return True
        if normalized.startswith(_LOCATION_REQUEST_PREFIXES):
            core = normalize_query(text, language)
            stripped = core if language.startswith("ar") else normalize_query(text, language)
            return not _strip_like_core(stripped, language)
        return False

    @staticmethod
    def _has_weak_campus_evidence(text: str, language: str) -> bool:
        normalized = normalize_query(text, language)
        if not normalized:
            return True
        core = _strip_like_core(normalized, language)
        if not core:
            return True
        tokens = core.split()
        if len(tokens) == 1 and len(tokens[0]) <= 2:
            return True
        return len(core) < 4

    def _trace_response(self, packet: ResponsePacket) -> None:
        self._trace(
            "response_generated",
            packet.session_id,
            text=packet.text,
            language=packet.language,
            should_navigate=packet.should_navigate,
        )

    def _trace(self, event_name: str, session_id: Optional[str], **fields) -> None:
        if self._trace_hook is None:
            return
        try:
            self._trace_hook(event_name, session_id=session_id, **fields)
        except Exception as exc:
            logger.debug("controller_trace_hook_error", event=event_name, error=str(exc))


def _strip_like_core(text: str, language: str) -> str:
    from app.retrieval.search import _strip_filler

    return _strip_filler(text, language)
