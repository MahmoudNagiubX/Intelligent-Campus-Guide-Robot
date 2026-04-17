"""
Navigator - Phase 2 Step 2.5: Router Tests
Tests for the intent routing service covering pre-rules, LLM classification,
JSON validation, target text extraction, and all failure paths.

Run with:
    pytest tests/unit/test_router.py -v

The Groq API is always mocked — no network calls are made in these tests.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.utils.contracts import IntentClass, IntentResult
from app.routing.router import route, _apply_pre_rules, _parse_router_response


# ─────────────────────────────────────────────────────────────────────────────
# Pre-rule tests (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestPreRules:
    """
    Pre-rules match fast keyword triggers before any LLM call.
    They return an IntentClass or None.
    """

    def test_navigation_take_me_to(self):
        assert _apply_pre_rules("take me to Lab 214") == IntentClass.NAVIGATION_REQUEST

    def test_navigation_guide_me_to(self):
        assert _apply_pre_rules("guide me to the library") == IntentClass.NAVIGATION_REQUEST

    def test_navigation_show_me_the_way(self):
        assert _apply_pre_rules("show me the way to Building C") == IntentClass.NAVIGATION_REQUEST

    def test_navigation_navigate_to(self):
        assert _apply_pre_rules("navigate to the Robotics Lab") == IntentClass.NAVIGATION_REQUEST

    def test_campus_where_is(self):
        assert _apply_pre_rules("where is Building C?") == IntentClass.CAMPUS_QUERY

    def test_campus_office_hours(self):
        assert _apply_pre_rules("what are the office hours of Dr Ahmed?") == IntentClass.CAMPUS_QUERY

    def test_campus_what_floor(self):
        assert _apply_pre_rules("what floor is the lab on?") == IntentClass.CAMPUS_QUERY

    def test_social_how_are_you(self):
        assert _apply_pre_rules("how are you?") == IntentClass.SOCIAL_CHAT

    def test_social_hello(self):
        assert _apply_pre_rules("hello") == IntentClass.SOCIAL_CHAT

    def test_social_good_morning(self):
        assert _apply_pre_rules("good morning") == IntentClass.SOCIAL_CHAT

    def test_no_match_returns_none(self):
        assert _apply_pre_rules("what is quantum mechanics?") is None

    def test_no_match_gibberish(self):
        assert _apply_pre_rules("asdfghjkl") is None

    def test_arabic_navigation_trigger(self):
        assert _apply_pre_rules("خدني للمعمل") == IntentClass.NAVIGATION_REQUEST

    def test_arabic_campus_trigger(self):
        assert _apply_pre_rules("فين المعمل؟") == IntentClass.CAMPUS_QUERY

    def test_case_insensitive(self):
        assert _apply_pre_rules("WHERE IS the library?") == IntentClass.CAMPUS_QUERY
        assert _apply_pre_rules("TAKE ME TO the gym") == IntentClass.NAVIGATION_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# JSON response parsing tests (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestParseRouterResponse:
    def _valid_json(self, **overrides) -> str:
        data = {
            "intent": "campus_query",
            "language": "en",
            "target_text": "Robotics Lab",
            "needs_clarification": False,
            "clarification_question": None,
            "confidence": 0.95,
            "reason": "user asked about a location",
        }
        data.update(overrides)
        return json.dumps(data)

    def test_parses_valid_campus_query(self):
        result = _parse_router_response(self._valid_json(), "where is the robotics lab")
        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.target_text == "Robotics Lab"
        assert result.confidence == 0.95
        assert result.language == "en"

    def test_parses_navigation_request(self):
        result = _parse_router_response(
            self._valid_json(intent="navigation_request"), "take me to lab 214"
        )
        assert result.intent == IntentClass.NAVIGATION_REQUEST

    def test_parses_social_chat(self):
        result = _parse_router_response(
            self._valid_json(intent="social_chat", target_text=None), "how are you"
        )
        assert result.intent == IntentClass.SOCIAL_CHAT
        assert result.target_text is None

    def test_parses_unknown(self):
        result = _parse_router_response(
            self._valid_json(intent="unknown", confidence=0.1), "quantum mechanics"
        )
        assert result.intent == IntentClass.UNKNOWN

    def test_invalid_json_returns_unknown(self):
        result = _parse_router_response("{not valid json}", "some text")
        assert result.intent == IntentClass.UNKNOWN

    def test_invalid_intent_string_returns_unknown(self):
        result = _parse_router_response(
            self._valid_json(intent="made_up_intent"), "test"
        )
        assert result.intent == IntentClass.UNKNOWN

    def test_confidence_clamped_to_0_to_1(self):
        result = _parse_router_response(self._valid_json(confidence=5.0), "test")
        assert result.confidence <= 1.0

    def test_null_string_target_text_converted_to_none(self):
        result = _parse_router_response(self._valid_json(target_text="null"), "test")
        assert result.target_text is None

    def test_empty_target_text_converted_to_none(self):
        result = _parse_router_response(self._valid_json(target_text=""), "test")
        assert result.target_text is None

    def test_arabic_language_preserved(self):
        result = _parse_router_response(
            self._valid_json(language="ar-EG"), "فين المعمل"
        )
        assert result.language == "ar-EG"

    def test_needs_clarification_true(self):
        result = _parse_router_response(
            self._valid_json(
                needs_clarification=True,
                clarification_question="Do you mean Lab A or Lab B?"
            ),
            "which lab"
        )
        assert result.needs_clarification is True
        assert result.clarification_question == "Do you mean Lab A or Lab B?"

    def test_raw_query_preserved(self):
        original = "where is building c"
        result = _parse_router_response(self._valid_json(), original)
        assert result.raw_query == original


# ─────────────────────────────────────────────────────────────────────────────
# Full routing service tests (LLM mocked)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_groq(response_json: str | None):
    """Return a mocked GroqClient.complete_json that returns the given string."""
    mock = MagicMock()
    mock.complete_json.return_value = response_json
    return mock


def _groq_response(**kwargs) -> str:
    defaults = {
        "intent": "campus_query",
        "language": "en",
        "target_text": None,
        "needs_clarification": False,
        "clarification_question": None,
        "confidence": 0.9,
        "reason": "test",
    }
    defaults.update(kwargs)
    return json.dumps(defaults)


class TestRouteService:
    """
    Tests for the high-level route() function.
    The Groq client is mocked so no real API calls are made.
    """

    def test_campus_query_classified(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="campus_query", target_text="Building C")
            )
            result = route("Where is Building C?")

        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.target_text == "Building C"

    def test_navigation_request_classified(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="navigation_request", target_text="Lab 214")
            )
            result = route("Take me to Lab 214")

        assert result.intent == IntentClass.NAVIGATION_REQUEST
        assert result.target_text == "Lab 214"

    def test_social_chat_classified(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="social_chat", confidence=0.97)
            )
            result = route("How are you?")

        assert result.intent == IntentClass.SOCIAL_CHAT

    def test_unknown_classified(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="unknown", confidence=0.2)
            )
            result = route("What is quantum mechanics?")

        assert result.intent == IntentClass.UNKNOWN

    def test_pre_rule_fires_before_llm_for_navigation(self):
        """Pre-rule must match before LLM is called for clear navigation phrases."""
        with patch("app.routing.router._get_groq") as mock_get_groq:
            # LLM says campus_query — but pre-rule should override to navigation_request
            mock_groq = _make_mock_groq(
                _groq_response(intent="campus_query", target_text="Library")
            )
            mock_get_groq.return_value = mock_groq
            result = route("take me to the library")

        # Pre-rule wins for intent, but LLM target_text is still used
        assert result.intent == IntentClass.NAVIGATION_REQUEST

    def test_pre_rule_fires_before_llm_for_campus(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_groq = _make_mock_groq(
                _groq_response(intent="social_chat")
            )
            mock_get_groq.return_value = mock_groq
            result = route("where is the library?")

        assert result.intent == IntentClass.CAMPUS_QUERY

    def test_llm_failure_with_pre_rule_returns_pre_rule_result(self):
        """If LLM fails but a pre-rule matched, use the pre-rule intent."""
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(None)  # LLM returns nothing
            result = route("take me to the library")

        assert result.intent == IntentClass.NAVIGATION_REQUEST
        assert result.confidence >= 0.7

    def test_llm_failure_no_pre_rule_returns_unknown(self):
        """If LLM fails and no pre-rule matched, return UNKNOWN."""
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(None)
            result = route("quantum mechanics lecture notes")

        assert result.intent == IntentClass.UNKNOWN

    def test_llm_exception_returns_unknown(self):
        """An exception from the LLM client must never propagate."""
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_groq = MagicMock()
            mock_groq.complete_json.side_effect = RuntimeError("connection refused")
            mock_get_groq.return_value = mock_groq
            result = route("quantum mechanics")

        assert result.intent == IntentClass.UNKNOWN

    def test_empty_transcript_returns_unknown(self):
        result = route("")
        assert result.intent == IntentClass.UNKNOWN

    def test_whitespace_only_transcript_returns_unknown(self):
        result = route("   ")
        assert result.intent == IntentClass.UNKNOWN

    def test_office_hours_query_classified_as_campus(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="campus_query", target_text="Dr Ahmed")
            )
            result = route("What are Dr Ahmed's office hours?")

        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.target_text == "Dr Ahmed"

    def test_tells_joke_classified_as_social(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="social_chat")
            )
            result = route("Tell me a joke")

        assert result.intent == IntentClass.SOCIAL_CHAT

    def test_arabic_office_hours_query(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="campus_query", language="ar-EG", target_text="Dr Ahmed")
            )
            result = route("ساعات العمل بتاعة دكتور أحمد")

        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.language in ("ar-EG", "en")  # pre-rule ساعات العمل covers this

    def test_result_always_has_raw_query(self):
        with patch("app.routing.router._get_groq") as mock_get_groq:
            mock_get_groq.return_value = _make_mock_groq(
                _groq_response(intent="social_chat")
            )
            result = route("hello navigator")

        assert result.raw_query == "hello navigator"
