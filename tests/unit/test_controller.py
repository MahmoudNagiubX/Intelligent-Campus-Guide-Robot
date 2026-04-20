from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from app.pipeline.controller import ConversationController
from app.pipeline.language_detector import LangResult
from app.utils.contracts import (
    IntentClass,
    IntentResult,
    ResponsePacket,
    RetrievalResult,
    RetrievalStatus,
    SpokenFacts,
    TranscriptEvent,
)


def _event(text: str, *, language: str = "en", language_confidence: float | None = None) -> TranscriptEvent:
    return TranscriptEvent(
        text=text,
        is_final=True,
        language=language,
        language_confidence=language_confidence,
        confidence=0.95,
        session_id="sess-1",
        source="deepgram",
    )


def _intent(intent: IntentClass, **overrides) -> IntentResult:
    payload = IntentResult(
        intent=intent,
        language="en",
        target_text="Robotics Lab",
        confidence=0.92,
        raw_query="where is the robotics lab",
    )
    return replace(payload, **overrides)


def _retrieval_ok(*, name: str = "Robotics Lab", nav_code: str | None = "NAV_C105") -> RetrievalResult:
    return RetrievalResult(
        status=RetrievalStatus.OK,
        entity_type="room",
        entity_id=1,
        canonical_name=name,
        spoken_facts=SpokenFacts(building="C", floor="1", room="C105", description="Lab"),
        nav_code=nav_code,
        confidence=0.95,
        matched_via="alias",
    )


class _GroqTextStub:
    def complete_text(self, *args, **kwargs) -> str:
        system_prompt = kwargs.get("system_prompt", "")
        return "المعمل في المبنى C." if any(ord(char) > 127 for char in system_prompt) else "The lab is in building C."


def test_controller_calls_search_directly_for_campus_queries() -> None:
    controller = ConversationController(groq=_GroqTextStub())

    with patch("app.pipeline.controller.detect_language", return_value=LangResult("en", "deepgram", 0.99)):
        with patch("app.pipeline.controller.route", return_value=_intent(IntentClass.CAMPUS_QUERY)):
            with patch("app.pipeline.controller.search", return_value=_retrieval_ok()) as search_mock:
                with patch("app.pipeline.response_composer._campus_prompt_for", return_value="{retrieval_facts}"):
                    result = controller.handle_transcript(_event("where is the robotics lab"))

    assert isinstance(result, ResponsePacket)
    assert result.language == "en"
    search_mock.assert_called_once_with("Robotics Lab", lang="en")


def test_controller_wires_detected_language_to_router_search_and_response_language() -> None:
    controller = ConversationController(groq=_GroqTextStub())

    with patch("app.pipeline.controller.detect_language", return_value=LangResult("ar-EG", "deepgram", 0.98)):
        with patch(
            "app.pipeline.controller.route",
            return_value=_intent(
                IntentClass.CAMPUS_QUERY,
                language="ar",
                target_text="معمل الروبوتات",
                raw_query="فين معمل الروبوتات",
            ),
        ) as route_mock:
            with patch(
                "app.pipeline.controller.search",
                return_value=RetrievalResult(
                    status=RetrievalStatus.OK,
                    entity_type="room",
                    entity_id=2,
                    canonical_name="معمل الروبوتات",
                    spoken_facts=SpokenFacts(building="C", floor="1", room="C105"),
                    nav_code="NAV_C105",
                    confidence=0.95,
                ),
            ) as search_mock:
                with patch("app.pipeline.response_composer._campus_prompt_for", return_value="{retrieval_facts}"):
                    result = controller.handle_transcript(
                        _event("فين معمل الروبوتات", language="ar-EG", language_confidence=0.97)
                    )

    route_mock.assert_called_once_with("فين معمل الروبوتات", lang_hint="ar-EG")
    search_mock.assert_called_once_with("معمل الروبوتات", lang="ar-EG")
    assert result.language == "ar-EG"


def test_controller_short_circuits_router_clarification() -> None:
    controller = ConversationController(groq=_GroqTextStub())

    with patch("app.pipeline.controller.detect_language", return_value=LangResult("en", "deepgram", 0.99)):
        with patch(
            "app.pipeline.controller.route",
            return_value=_intent(
                IntentClass.CAMPUS_QUERY,
                target_text=None,
                needs_clarification=True,
                clarification_question="Do you mean the robotics lab or the robotics office?",
            ),
        ):
            with patch("app.pipeline.controller.search") as search_mock:
                result = controller.handle_transcript(_event("where is robotics"))

    assert "robotics" in result.text.lower()
    search_mock.assert_not_called()


def test_controller_navigation_requires_trusted_nav_code() -> None:
    controller = ConversationController(groq=_GroqTextStub())

    with patch("app.pipeline.controller.detect_language", return_value=LangResult("en", "deepgram", 0.99)):
        with patch(
            "app.pipeline.controller.route",
            return_value=_intent(IntentClass.NAVIGATION_REQUEST, target_text="Robotics Lab"),
        ):
            with patch("app.pipeline.controller.search", return_value=_retrieval_ok(nav_code=None)):
                with patch("app.pipeline.response_composer._campus_prompt_for", return_value="{retrieval_facts}"):
                    result = controller.handle_transcript(_event("take me to the robotics lab"))

    assert result.should_navigate is False
    assert "trusted navigation route" in result.text.lower()


def test_controller_social_path_keeps_language_and_skips_search() -> None:
    controller = ConversationController(groq=_GroqTextStub())

    with patch("app.pipeline.controller.detect_language", return_value=LangResult("ar-EG", "unicode_heuristic", 0.9)):
        with patch(
            "app.pipeline.controller.route",
            return_value=_intent(IntentClass.SOCIAL_CHAT, language="ar", target_text=None, raw_query="ازيك"),
        ):
            with patch("app.pipeline.controller.search") as search_mock:
                with patch("app.pipeline.response_composer._load_social_prompt", return_value="reply in same language"):
                    result = controller.handle_transcript(_event("ازيك", language="ar-EG"))

    assert result.language == "ar-EG"
    search_mock.assert_not_called()
