"""
Navigator - Phase 5+6 Tests: Controller and Response Composer
Tests for ConversationController and ResponseComposer.

All LLM calls and retrieval calls are mocked.
No network access, no API keys, no real DB required.

Run with:
    pytest tests/unit/test_controller.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.controller import ConversationController
from app.pipeline.response_composer import ResponseComposer, _load_campus_prompt, _load_social_prompt
from app.utils.contracts import (
    IntentClass,
    IntentResult,
    NavigationCommand,
    ResponsePacket,
    RetrievalResult,
    RetrievalStatus,
    SpokenFacts,
    TranscriptEvent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_groq(spoken_text: str = "Test response.") -> MagicMock:
    groq = MagicMock()
    groq.complete_json.return_value = spoken_text
    return groq


def _make_event(
    text: str,
    language: str = "en",
    session_id: str = "sess-001",
) -> TranscriptEvent:
    return TranscriptEvent(
        text=text,
        is_final=True,
        language=language,
        confidence=0.95,
        session_id=session_id,
        source="deepgram_mock",
    )


def _make_intent(
    intent: IntentClass,
    target_text: str = None,
    language: str = "en",
    needs_clarification: bool = False,
    clarification_question: str = None,
    confidence: float = 0.9,
    raw_query: str = "test query",
) -> IntentResult:
    return IntentResult(
        intent=intent,
        language=language,
        target_text=target_text,
        needs_clarification=needs_clarification,
        clarification_question=clarification_question,
        confidence=confidence,
        raw_query=raw_query,
    )


def _make_retrieval_ok(
    name: str = "Robotics Lab",
    entity_type: str = "location",
    nav_code: str = "NAV_LAB_214",
    building: str = "Building C",
    floor: str = "2",
    room: str = "214",
) -> RetrievalResult:
    return RetrievalResult(
        status=RetrievalStatus.OK,
        canonical_name=name,
        entity_type=entity_type,
        confidence=0.95,
        matched_via="alias",
        nav_code=nav_code,
        map_node="c2_r214",
        spoken_facts=SpokenFacts(
            building=building,
            floor=floor,
            room=room,
            description="Main robotics lab",
        ),
    )


def _make_retrieval_not_found() -> RetrievalResult:
    return RetrievalResult(status=RetrievalStatus.NOT_FOUND)


def _make_retrieval_ambiguous() -> RetrievalResult:
    return RetrievalResult(
        status=RetrievalStatus.AMBIGUOUS,
        candidates=["Robotics Lab", "Robotics Office"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# ResponseComposer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseComposerCampusPath:
    def test_ok_retrieval_calls_llm_and_returns_spoken(self):
        groq = _mock_groq("The Robotics Lab is in Building C, second floor.")
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_campus_prompt", return_value="sys prompt"):
            result = composer.compose_campus_answer(
                _make_retrieval_ok(), "where is the robotics lab", "en", "s1"
            )
        assert isinstance(result, ResponsePacket)
        assert "Robotics Lab" in result.text or "Building C" in result.text
        assert result.language == "en"
        assert result.session_id == "s1"

    def test_not_found_returns_fallback_without_llm_call(self):
        groq = _mock_groq("This should not appear")
        composer = ResponseComposer(groq=groq)
        result = composer.compose_campus_answer(
            _make_retrieval_not_found(), "quantum mechanics", "en", "s1"
        )
        assert "database" in result.text.lower() or "find" in result.text.lower()
        groq.complete_json.assert_not_called()

    def test_ambiguous_returns_clarification_question(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_campus_answer(
            _make_retrieval_ambiguous(), "robotics", "en", "s1"
        )
        assert "Robotics Lab" in result.text or "match" in result.text.lower()

    def test_arabic_not_found_uses_arabic_fallback(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_campus_answer(
            _make_retrieval_not_found(), "test", "ar-EG", "s1"
        )
        # Arabic fallback contains Arabic text
        assert any(ord(c) > 127 for c in result.text)

    def test_llm_error_falls_back_to_facts(self):
        groq = MagicMock()
        groq.complete_json.side_effect = RuntimeError("timeout")
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_campus_prompt", return_value="p"):
            result = composer.compose_campus_answer(
                _make_retrieval_ok(), "where is lab", "en", "s1"
            )
        # Should fallback to plain text from facts
        assert "Robotics Lab" in result.text or "Building C" in result.text

    def test_llm_returns_none_falls_back(self):
        groq = _mock_groq(None)
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_campus_prompt", return_value="p"):
            result = composer.compose_campus_answer(
                _make_retrieval_ok(), "where is lab", "en", "s1"
            )
        assert isinstance(result.text, str)
        assert len(result.text) > 0


class TestResponseComposerNavigationPath:
    def test_ok_with_nav_code_emits_navigation_command(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_navigation_answer(
            _make_retrieval_ok(), "take me to robotics lab", "en", "s1"
        )
        assert result.should_navigate is True
        assert isinstance(result.navigation_command, NavigationCommand)
        assert result.navigation_command.target_code == "NAV_LAB_214"
        assert result.navigation_command.target_label == "Robotics Lab"

    def test_ok_without_nav_code_does_not_navigate(self):
        groq = _mock_groq("The library is on floor 1.")
        composer = ResponseComposer(groq=groq)
        retrieval = _make_retrieval_ok(nav_code=None, name="Main Library")
        with patch("app.pipeline.response_composer._load_campus_prompt", return_value="p"):
            result = composer.compose_navigation_answer(retrieval, "take me to library", "en", "s1")
        assert result.should_navigate is False
        assert "route" in result.text.lower() or "yet" in result.text.lower()

    def test_not_found_blocks_navigation(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_navigation_answer(
            _make_retrieval_not_found(), "take me there", "en", "s1"
        )
        assert result.should_navigate is False

    def test_ambiguous_blocks_navigation_with_question(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_navigation_answer(
            _make_retrieval_ambiguous(), "take me to lab", "en", "s1"
        )
        assert result.should_navigate is False

    def test_arabic_navigation_confirmation(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_navigation_answer(
            _make_retrieval_ok(), "خدني للمعمل", "ar-EG", "s1"
        )
        assert result.should_navigate is True
        # Arabic confirmation should contain Arabic script
        assert any(ord(c) > 127 for c in result.text)


class TestResponseComposerSocialPath:
    def test_social_path_calls_llm(self):
        groq = _mock_groq("Hey there! I'm Navigator, your campus guide.")
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_social_prompt", return_value="p"):
            result = composer.compose_social_answer("how are you", "en", "s1")
        assert result.language == "en"
        assert isinstance(result.text, str)
        groq.complete_json.assert_called_once()

    def test_social_llm_error_falls_back_gracefully(self):
        groq = MagicMock()
        groq.complete_json.side_effect = RuntimeError("timeout")
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_social_prompt", return_value="p"):
            result = composer.compose_social_answer("hello", "en", "s1")
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_social_arabic_fallback_on_error(self):
        groq = MagicMock()
        groq.complete_json.side_effect = RuntimeError("x")
        composer = ResponseComposer(groq=groq)
        with patch("app.pipeline.response_composer._load_social_prompt", return_value="p"):
            result = composer.compose_social_answer("ازيك", "ar-EG", "s1")
        assert any(ord(c) > 127 for c in result.text)


class TestResponseComposerUnknownPath:
    def test_unknown_no_llm_call(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_unknown_answer("en", "s1")
        assert isinstance(result.text, str)
        assert "campus" in result.text.lower()
        groq.complete_json.assert_not_called()

    def test_unknown_arabic(self):
        groq = _mock_groq()
        composer = ResponseComposer(groq=groq)
        result = composer.compose_unknown_answer("ar-EG", "s1")
        assert any(ord(c) > 127 for c in result.text)


class TestResponseComposerClean:
    def test_clean_spoken_strips_json_wrapper(self):
        raw = json.dumps({"text": "The lab is on floor 2."})
        assert ResponseComposer._clean_spoken(raw) == "The lab is on floor 2."

    def test_clean_spoken_strips_markdown(self):
        raw = "```\nThe lab is on floor 2.\n```"
        assert "The lab is on floor 2." in ResponseComposer._clean_spoken(raw)

    def test_clean_spoken_plain_text(self):
        raw = "The lab is on floor 2."
        assert ResponseComposer._clean_spoken(raw) == raw


# ─────────────────────────────────────────────────────────────────────────────
# ConversationController Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationController:
    """All routing and retrieval is mocked."""

    def _make_controller(self, groq_spoken: str = "Test.") -> ConversationController:
        ctrl = ConversationController(groq=_mock_groq(groq_spoken))
        return ctrl

    def _mock_route(self, intent: IntentClass, target: str = None, **kw):
        """Return a context manager that patches route()."""
        return patch(
            "app.pipeline.controller.route",
            return_value=_make_intent(intent, target_text=target, **kw),
        )

    def _mock_retrieve(self, result: RetrievalResult):
        return patch("app.pipeline.controller.retrieve", return_value=result)

    # Social path
    def test_social_chat_returns_response(self):
        ctrl = self._make_controller("Hey there!")
        with self._mock_route(IntentClass.SOCIAL_CHAT):
            with patch("app.pipeline.response_composer._load_social_prompt", return_value="p"):
                result = ctrl.handle_transcript(_make_event("how are you"))
        assert isinstance(result, ResponsePacket)
        assert result.language == "en"

    # Unknown path
    def test_unknown_returns_bounded_fallback(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.UNKNOWN):
            result = ctrl.handle_transcript(_make_event("quantum mechanics"))
        assert "campus" in result.text.lower() or "location" in result.text.lower()
        assert result.should_navigate is False

    # Campus path
    def test_campus_query_calls_retrieve(self):
        ctrl = self._make_controller("The Robotics Lab is on second floor.")
        with self._mock_route(IntentClass.CAMPUS_QUERY, target="Robotics Lab"):
            with self._mock_retrieve(_make_retrieval_ok()):
                with patch("app.pipeline.response_composer._load_campus_prompt", return_value="p"):
                    result = ctrl.handle_transcript(_make_event("where is the robotics lab"))
        assert isinstance(result, ResponsePacket)
        assert result.should_navigate is False

    def test_campus_not_found_returns_fallback(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.CAMPUS_QUERY, target="nonexistent"):
            with self._mock_retrieve(_make_retrieval_not_found()):
                result = ctrl.handle_transcript(_make_event("quantum mechanics"))
        assert "database" in result.text.lower() or "find" in result.text.lower()

    # Navigation path
    def test_navigation_with_valid_target_emits_command(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.NAVIGATION_REQUEST, target="Lab 214"):
            with self._mock_retrieve(_make_retrieval_ok()):
                result = ctrl.handle_transcript(_make_event("take me to lab 214"))
        assert result.should_navigate is True
        assert result.navigation_command is not None
        assert result.navigation_command.target_code == "NAV_LAB_214"

    def test_navigation_ambiguous_does_not_navigate(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.NAVIGATION_REQUEST, target="lab"):
            with self._mock_retrieve(_make_retrieval_ambiguous()):
                result = ctrl.handle_transcript(_make_event("take me to a lab"))
        assert result.should_navigate is False

    def test_navigation_not_found_does_not_navigate(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.NAVIGATION_REQUEST, target="nowhere"):
            with self._mock_retrieve(_make_retrieval_not_found()):
                result = ctrl.handle_transcript(_make_event("take me to nowhere"))
        assert result.should_navigate is False

    # Clarification
    def test_router_clarification_returned_without_retrieval(self):
        ctrl = self._make_controller()
        with patch(
            "app.pipeline.controller.route",
            return_value=_make_intent(
                IntentClass.CAMPUS_QUERY,
                needs_clarification=True,
                clarification_question="Do you mean Lab A or Lab B?",
            )
        ):
            result = ctrl.handle_transcript(_make_event("which lab"))
        assert "Lab A" in result.text or "Lab B" in result.text
        assert result.should_navigate is False

    # Edge cases
    def test_empty_transcript_returns_unknown(self):
        ctrl = self._make_controller()
        result = ctrl.handle_transcript(_make_event(""))
        assert isinstance(result, ResponsePacket)
        assert result.should_navigate is False

    def test_whitespace_transcript_returns_unknown(self):
        ctrl = self._make_controller()
        result = ctrl.handle_transcript(_make_event("   "))
        assert isinstance(result, ResponsePacket)

    def test_router_exception_returns_safe_packet(self):
        ctrl = self._make_controller()
        with patch("app.pipeline.controller.route", side_effect=RuntimeError("crash")):
            result = ctrl.handle_transcript(_make_event("test"))
        assert isinstance(result, ResponsePacket)
        assert result.should_navigate is False

    def test_session_id_preserved_in_response(self):
        ctrl = self._make_controller()
        with self._mock_route(IntentClass.UNKNOWN):
            result = ctrl.handle_transcript(_make_event("hello", session_id="abc-123"))
        assert result.session_id == "abc-123"

    def test_arabic_transcript_returns_arabic_response(self):
        ctrl = self._make_controller("مرحباً!")
        with self._mock_route(IntentClass.SOCIAL_CHAT, language="ar-EG"):
            with patch("app.pipeline.response_composer._load_social_prompt", return_value="p"):
                result = ctrl.handle_transcript(_make_event("ازيك", language="ar-EG"))
        assert result.language == "ar-EG"

    def test_result_is_always_response_packet(self):
        """No matter what, the controller always returns a ResponsePacket."""
        ctrl = self._make_controller()
        with patch("app.pipeline.controller.route", side_effect=Exception("total failure")):
            result = ctrl.handle_transcript(_make_event("test"))
        assert isinstance(result, ResponsePacket)
