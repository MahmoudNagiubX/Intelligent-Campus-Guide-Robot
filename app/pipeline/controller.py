"""
Navigator - Dual-Path Conversation Controller

Receives final transcripts, detects language, classifies intent, dispatches to
retrieval or social handling, and returns a ResponsePacket for TTS/actions.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.llm.groq_client import GroqClient
from app.pipeline.arabic_normalizer import normalize_arabic_transcript, normalize_room_reference
from app.pipeline.language_detector import LangResult, detect_language, lang_is_arabic
from app.pipeline.query_understander import understand
from app.pipeline.response_composer import ResponseComposer
from app.retrieval.hybrid_retriever import HybridResult, retrieve_hybrid
from app.retrieval.search import normalize_query, search
from app.routing.router import route
from app.stt.dual_stt_client import _looks_like_phonetic_arabic
from app.utils.contracts import IntentClass, IntentResult, ResponsePacket, RetrievalResult, TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_PREFLIGHT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db_preflight")
_TRANSCRIPT_CORRECTIONS = {
    "robotic slab": "robotics lab",
    "robot room": "robotics lab",
}
_NOISE_WORDS = frozenset({"um", "uh", "hmm", "hm", "ah", "er", "mm", "oh"})
_LATENCY_BUDGETS_MS = {
    "stt_to_router": 50,
    "router": 500,
    "retrieval": 50,
    "composer": 600,
    "tts": 400,
    "total": 1500,
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
        raw_text = event.text or ""
        text = raw_text.strip()
        cfg = get_settings()
        turn_started = time.monotonic()
        if len(text) < 2 or _is_noise_transcript(raw_text):
            logger.debug("controller.transcript_skipped_noise", text=repr(raw_text), session_id=session_id)
            return ResponsePacket(text="", language=cfg.default_language, session_id=session_id)

        detected_lang = detect_language(
            text=raw_text,
            deepgram_lang=event.language,
            deepgram_confidence=event.language_confidence,
            confidence_threshold=cfg.lang_confidence_threshold,
            default_lang=cfg.default_language,
        )
        if detected_lang.source == "stt_provider" and detected_lang.code == "en":
            if event.source.startswith("deepgram") and _looks_like_phonetic_arabic(raw_text):
                detected_lang = LangResult(code="ar", source="phonetic_heuristic", confidence=0.75)
                logger.warning(
                    "controller.lang_override_phonetic_arabic",
                    original_text=raw_text[:60],
                    session_id=session_id,
                )
        language = detected_lang.code
        logger.info(
            "controller.lang_detected",
            code=language,
            source=detected_lang.source,
            confidence=detected_lang.confidence,
            stt_source=event.source,
            session_id=session_id,
        )

        if lang_is_arabic(detected_lang):
            routing_text = normalize_room_reference(normalize_arabic_transcript(raw_text))
        else:
            routing_text = normalize_room_reference(_apply_en_corrections(raw_text))

        logger.info(
            "controller.transcript_normalized",
            raw=raw_text[:80],
            normalized=routing_text[:80],
            lang=language,
            session_id=session_id,
        )

        logger.info(
            "controller_turn_start",
            text=raw_text[:80],
            language=language,
            language_source=detected_lang.source,
            session_id=session_id,
        )

        if not text:
            packet = self._composer.compose_unknown_answer(language=language, session_id=session_id)
            self._trace_response(packet)
            return packet

        pre_router_packet = self._pre_router_quality_gate(routing_text, language, session_id)
        if pre_router_packet is not None:
            self._trace_response(pre_router_packet)
            return pre_router_packet

        try:
            _check_latency("stt_to_router", (time.monotonic() - turn_started) * 1000)
            packet = self._dispatch(
                text=routing_text,
                raw_text=raw_text,
                language=language,
                session_id=session_id,
                stt_confidence=event.confidence,
                detected_language=detected_lang.code,
            )
            _check_latency("total", (time.monotonic() - turn_started) * 1000)
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
        raw_text: str,
        language: str,
        session_id: Optional[str],
        stt_confidence: float,
        detected_language: str,
    ) -> ResponsePacket:
        router_started = time.monotonic()
        intent_result = self._classify(text, detected_language, session_id)
        _check_latency("router", (time.monotonic() - router_started) * 1000)

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
            return self._handle_campus_query(intent_result, raw_text, language, session_id, stt_confidence)
        if intent_result.intent == IntentClass.NAVIGATION_REQUEST:
            return self._handle_navigation_request(intent_result, raw_text, language, session_id, stt_confidence)
        if intent_result.intent == IntentClass.SOCIAL_CHAT:
            return self._handle_social_chat(raw_text, language, session_id)
        if language.startswith("ar"):
            return self._composer.compose_unknown_answer(language=language, session_id=session_id)
        return self._composer.compose_general_campus_answer(raw_text, language=language, session_id=session_id)

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
        retrieval_started = time.monotonic()
        if not language.startswith("ar"):
            hybrid = self._resolve_hybrid_result(intent_result, original_query, language, session_id)
            _check_latency("retrieval", (time.monotonic() - retrieval_started) * 1000)
            self._trace_hybrid_result("campus", session_id, hybrid)
            return self._compose_hybrid_campus_answer(hybrid, original_query, language, session_id)

        query, retrieval = self._resolve_retrieval_query(query, stt_confidence, language, session_id)
        _check_latency("retrieval", (time.monotonic() - retrieval_started) * 1000)
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

        composer_started = time.monotonic()
        packet = self._composer.compose_campus_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )
        _check_latency("composer", (time.monotonic() - composer_started) * 1000)
        return packet

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
        retrieval_started = time.monotonic()
        if not language.startswith("ar"):
            hybrid = self._resolve_hybrid_result(intent_result, original_query, language, session_id)
            _check_latency("retrieval", (time.monotonic() - retrieval_started) * 1000)
            self._trace_hybrid_result("navigation", session_id, hybrid)
            return self._compose_hybrid_navigation_answer(hybrid, original_query, language, session_id)

        query, retrieval = self._resolve_retrieval_query(query, stt_confidence, language, session_id)
        _check_latency("retrieval", (time.monotonic() - retrieval_started) * 1000)
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

        composer_started = time.monotonic()
        packet = self._composer.compose_navigation_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )
        _check_latency("composer", (time.monotonic() - composer_started) * 1000)
        return packet

    def _resolve_hybrid_result(
        self,
        intent_result: IntentResult,
        original_query: str,
        language: str,
        session_id: Optional[str],
    ) -> HybridResult:
        understood = understand(
            raw_query=original_query,
            router_entity=intent_result.target_text or "",
            router_confidence=intent_result.confidence,
        )
        future: Future[HybridResult] | None = None
        try:
            future = _PREFLIGHT_EXECUTOR.submit(retrieve_hybrid, understood, language)
            return future.result(timeout=2.0)
        except FutureTimeoutError:
            logger.warning("controller.hybrid_preflight_timeout", session_id=session_id)
        except Exception as exc:
            logger.warning("controller.hybrid_preflight_failed", error=str(exc), session_id=session_id)
        if future is not None:
            future.cancel()
        return retrieve_hybrid(understood, language)

    def _compose_hybrid_campus_answer(
        self,
        hybrid: HybridResult,
        original_query: str,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        composer_started = time.monotonic()
        packet: ResponsePacket
        if hybrid.answered_by == "db" and hybrid.db_result is not None:
            packet = self._composer.compose_campus_answer(hybrid.db_result, original_query, language, session_id)
        elif hybrid.answered_by == "clarification" and hybrid.db_result is not None:
            packet = self._composer.compose_campus_answer(hybrid.db_result, original_query, language, session_id)
        elif hybrid.answered_by == "ecu_web" and hybrid.ecu_result is not None:
            packet = self._composer.compose_ecu_answer(hybrid.ecu_result, original_query, language, session_id)
        else:
            packet = self._composer.compose_general_campus_answer(original_query, language, session_id)
        _check_latency("composer", (time.monotonic() - composer_started) * 1000)
        return packet

    def _compose_hybrid_navigation_answer(
        self,
        hybrid: HybridResult,
        original_query: str,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        composer_started = time.monotonic()
        if hybrid.answered_by == "db" and hybrid.db_result is not None:
            packet = self._composer.compose_navigation_answer(hybrid.db_result, original_query, language, session_id)
        elif hybrid.answered_by == "clarification" and hybrid.db_result is not None:
            packet = self._composer.compose_navigation_answer(hybrid.db_result, original_query, language, session_id)
        elif hybrid.answered_by == "ecu_web" and hybrid.ecu_result is not None:
            packet = self._composer.compose_ecu_answer(hybrid.ecu_result, original_query, language, session_id)
        else:
            packet = self._composer.compose_general_campus_answer(original_query, language, session_id)
        _check_latency("composer", (time.monotonic() - composer_started) * 1000)
        return packet

    def _trace_hybrid_result(self, path: str, session_id: Optional[str], hybrid: HybridResult) -> None:
        db_result = hybrid.db_result
        self._trace(
            "retrieval_finished",
            session_id,
            path=path,
            status=db_result.status.value if db_result else hybrid.answered_by,
            entity=db_result.canonical_name if db_result else (hybrid.ecu_result.title if hybrid.ecu_result else None),
            answered_by=hybrid.answered_by,
            nav_code=db_result.nav_code if db_result else None,
        )

    def _handle_social_chat(self, text: str, language: str, session_id: Optional[str]) -> ResponsePacket:
        composer_started = time.monotonic()
        packet = self._composer.compose_social_answer(transcript=text, language=language, session_id=session_id)
        _check_latency("composer", (time.monotonic() - composer_started) * 1000)
        return packet

    def _classify(self, text: str, language: str, session_id: Optional[str]) -> IntentResult:
        try:
            return route(text, lang_hint=language)
        except TimeoutError:
            logger.warning("controller.router_timeout", session_id=session_id)
            return IntentResult(
                intent=IntentClass.UNKNOWN,
                language=language,
                raw_query=text,
                reason="timeout",
            )
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
            logger.debug("controller_trace_hook_error", trace_event=event_name, error=str(exc))


def _strip_like_core(text: str, language: str) -> str:
    from app.retrieval.search import _strip_filler

    return _strip_filler(text, language)


@lru_cache(maxsize=1)
def _load_en_corrections() -> dict[str, str]:
    """Load English STT corrections from data/corrections_en.json."""
    path = Path("data/corrections_en.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def _apply_en_corrections(text: str) -> str:
    """Apply configured English transcript corrections before routing."""
    corrected = (text or "").lower()
    for wrong, right in _load_en_corrections().items():
        corrected = corrected.replace(wrong, right)
    return corrected


def _is_noise_transcript(text: str) -> bool:
    """Return True for short filler, punctuation-only, or single-letter noise."""
    stripped = (text or "").strip()
    if len(stripped) < 2:
        return True
    if stripped.lower() in _NOISE_WORDS:
        return True
    if len(stripped) == 1 and "\u0600" <= stripped <= "\u06FF":
        return True
    non_punct = sum(1 for char in stripped if char.isalnum() or "\u0600" <= char <= "\u06FF")
    return bool(stripped) and (non_punct / len(stripped)) < 0.20


def _check_latency(stage: str, elapsed_ms: float) -> None:
    """Warn when a hot-path stage exceeds its latency budget."""
    budget = _LATENCY_BUDGETS_MS.get(stage)
    if budget and elapsed_ms > budget:
        logger.warning(
            "controller.latency_budget_exceeded",
            stage=stage,
            budget_ms=budget,
            actual_ms=round(elapsed_ms),
        )
