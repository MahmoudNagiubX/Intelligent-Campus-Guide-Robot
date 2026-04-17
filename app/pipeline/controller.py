"""
Navigator - Dual-Path Conversation Controller
Phase 5, Step 5.1

The central hub that receives final transcripts, classifies intent,
dispatches to the correct path, and returns a ResponsePacket for TTS + actions.

Architecture:
    TranscriptEvent -> route() -> one of four paths -> ResponsePacket

Paths:
    campus_query        -> retrieve() -> compose_campus_answer()
    navigation_request  -> retrieve() -> compose_navigation_answer()
    social_chat         -> compose_social_answer()
    unknown             -> compose_unknown_answer()   (no LLM call)

If the router signals needs_clarification, the clarification path fires
before any retrieval.

All errors are caught at this layer — the controller never propagates
exceptions to callers.
"""

from __future__ import annotations

from typing import Optional

from app.llm.groq_client import GroqClient
from app.pipeline.response_composer import ResponseComposer
from app.retrieval.search import retrieve
from app.routing.router import route
from app.utils.contracts import (
    IntentClass,
    IntentResult,
    ResponsePacket,
    RetrievalStatus,
    TranscriptEvent,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationController:
    """
    Orchestrates one full conversation turn from raw transcript to spoken response.

    Args:
        groq: Shared GroqClient instance (used by both router and response composer).
    """

    def __init__(self, groq: Optional[GroqClient] = None) -> None:
        self._groq     = groq or GroqClient()
        self._composer = ResponseComposer(groq=self._groq)

    # ── Main entry point ──────────────────────────────────────────────────────

    def handle_transcript(self, event: TranscriptEvent) -> ResponsePacket:
        """
        Process one final transcript event and return a ResponsePacket.

        This is the ONLY public method callers should use.
        Never raises — always returns a safe ResponsePacket.

        Args:
            event: A validated TranscriptEvent with is_final=True.

        Returns:
            ResponsePacket ready for TTS and (optionally) the action bridge.
        """
        session_id = event.session_id
        language   = event.language or "en"
        text       = (event.text or "").strip()

        logger.info(
            "controller_turn_start",
            text=text[:80],
            language=language,
            session_id=session_id,
        )

        if not text:
            return self._composer.compose_unknown_answer(language=language, session_id=session_id)

        try:
            return self._dispatch(text, language, session_id)
        except Exception as exc:
            logger.error("controller_unhandled_error", error=str(exc), session_id=session_id)
            return ResponsePacket(
                text="Something went wrong. Please try again.",
                language=language,
                session_id=session_id,
            )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        text: str,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        """Classify intent and call the appropriate path handler."""
        intent_result = self._classify(text, session_id)

        logger.info(
            "controller_intent",
            intent=intent_result.intent.value,
            target=intent_result.target_text,
            confidence=round(intent_result.confidence, 2),
            needs_clarification=intent_result.needs_clarification,
            session_id=session_id,
        )

        # Router-signaled clarification — ask before retrieving
        if intent_result.needs_clarification and intent_result.clarification_question:
            return ResponsePacket(
                text=intent_result.clarification_question,
                language=language,
                session_id=session_id,
            )

        match intent_result.intent:
            case IntentClass.CAMPUS_QUERY:
                return self._handle_campus_query(intent_result, language, session_id)
            case IntentClass.NAVIGATION_REQUEST:
                return self._handle_navigation_request(intent_result, language, session_id)
            case IntentClass.SOCIAL_CHAT:
                return self._handle_social_chat(text, language, session_id)
            case _:
                return self._composer.compose_unknown_answer(language=language, session_id=session_id)

    # ── Campus path (§5.2) ────────────────────────────────────────────────────

    def _handle_campus_query(
        self,
        intent_result: IntentResult,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        """Retrieve campus facts and compose a spoken answer."""
        query = intent_result.target_text or intent_result.raw_query or ""
        original_query = intent_result.raw_query or ""

        logger.info("controller_campus_retrieve", query=query, session_id=session_id)
        retrieval = retrieve(query)

        logger.info(
            "controller_campus_retrieval_done",
            status=retrieval.status.value,
            entity=retrieval.canonical_name,
            session_id=session_id,
        )

        return self._composer.compose_campus_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )

    # ── Navigation path (§5.3) ────────────────────────────────────────────────

    def _handle_navigation_request(
        self,
        intent_result: IntentResult,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        """Retrieve the navigation target and emit an action command if safe."""
        query = intent_result.target_text or intent_result.raw_query or ""
        original_query = intent_result.raw_query or ""

        logger.info("controller_nav_retrieve", query=query, session_id=session_id)
        retrieval = retrieve(query)

        logger.info(
            "controller_nav_retrieval_done",
            status=retrieval.status.value,
            entity=retrieval.canonical_name,
            nav_code=retrieval.nav_code,
            session_id=session_id,
        )

        return self._composer.compose_navigation_answer(
            retrieval=retrieval,
            original_query=original_query,
            language=language,
            session_id=session_id,
        )

    # ── Social path (§5.4) ────────────────────────────────────────────────────

    def _handle_social_chat(
        self,
        text: str,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        """Generate a warm, short social chat response."""
        return self._composer.compose_social_answer(
            transcript=text,
            language=language,
            session_id=session_id,
        )

    # ── Internal classification ────────────────────────────────────────────────

    def _classify(self, text: str, session_id: Optional[str]) -> IntentResult:
        """Call the router and return an IntentResult. Never raises."""
        try:
            return route(text)
        except Exception as exc:
            logger.error("controller_router_error", error=str(exc), session_id=session_id)
            from app.utils.contracts import IntentClass
            return IntentResult(
                intent=IntentClass.UNKNOWN,
                language="en",
                raw_query=text,
                reason="router_exception",
            )
