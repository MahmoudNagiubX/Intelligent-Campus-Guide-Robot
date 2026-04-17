"""
Unit tests for the Groq LLM client wrapper — Phase 2, Step 2.1.

All Groq API calls are mocked.  No network access, no API key required.

Test groups:
- TestRouterRawOutput  — Pydantic model validation and conversion
- TestCompleteJson     — low-level retry/timeout/backoff behaviour
- TestCallRouter       — high-level classification and failure paths
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from app.llm.groq_client import GroqClient, _unknown_result
from app.llm.models import RouterRawOutput, parse_router_response
from app.utils.contracts import IntentClass, IntentResult


# ─────────────────────────────────────────────────────────────────────────────
# Shared test helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_groq_response(content: str) -> MagicMock:
    """Build a minimal mock that matches the Groq completion response shape."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.usage.total_tokens = 42
    return response


def _make_timeout_error() -> APITimeoutError:
    """Construct a minimal APITimeoutError (requires an httpx.Request)."""
    return APITimeoutError(request=httpx.Request("POST", "https://api.groq.com"))


def _make_connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "https://api.groq.com"),
        message="connection refused",
    )


SYSTEM_PROMPT = "Classify the intent. Return JSON only."
TRANSCRIPT = "where is the robotics lab"


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.has_groq_key = True
    s.groq_api_key = "test-key-abc"
    s.groq_model = "llama-3.1-8b-instant"
    s.groq_timeout = 8.0
    s.groq_max_retries = 3
    s.groq_retry_backoff = 0.0  # no real sleep in tests
    return s


@pytest.fixture
def groq_client(mock_settings: MagicMock) -> GroqClient:
    """
    GroqClient with mocked settings and a fully mocked SDK client.
    Tests set groq_client._client.chat.completions.create as needed.
    """
    with (
        patch("app.llm.groq_client.get_settings", return_value=mock_settings),
        patch("app.llm.groq_client.Groq"),  # prevent real SDK init
    ):
        client = GroqClient()

    client._client = MagicMock()
    return client


# ─────────────────────────────────────────────────────────────────────────────
# TestRouterRawOutput — model validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRouterRawOutput:
    def test_valid_campus_query(self) -> None:
        model = RouterRawOutput.model_validate({
            "intent": "campus_query",
            "language": "en",
            "target_text": "Robotics Lab",
            "needs_clarification": False,
            "confidence": 0.92,
            "reason": "user asked about location",
        })
        assert model.intent == "campus_query"
        assert model.target_text == "Robotics Lab"
        assert model.confidence == 0.92

    def test_valid_navigation_request_uppercase(self) -> None:
        """Uppercase intent values must be normalized and accepted."""
        model = RouterRawOutput.model_validate({"intent": "Navigation_Request"})
        assert model.intent == "navigation_request"

    def test_valid_social_chat(self) -> None:
        model = RouterRawOutput.model_validate({"intent": "social_chat"})
        assert model.intent == "social_chat"

    def test_valid_unknown(self) -> None:
        model = RouterRawOutput.model_validate({"intent": "unknown"})
        assert model.intent == "unknown"

    def test_invalid_intent_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RouterRawOutput.model_validate({"intent": "launch_rockets"})

    def test_confidence_clamped_above_one(self) -> None:
        model = RouterRawOutput.model_validate({"intent": "unknown", "confidence": 5.0})
        assert model.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        model = RouterRawOutput.model_validate({"intent": "unknown", "confidence": -0.5})
        assert model.confidence == 0.0

    def test_non_numeric_confidence_defaults_to_zero(self) -> None:
        model = RouterRawOutput.model_validate({"intent": "unknown", "confidence": "high"})
        assert model.confidence == 0.0

    def test_to_intent_result_conversion(self) -> None:
        result = RouterRawOutput.model_validate({
            "intent": "campus_query",
            "language": "ar-EG",
            "target_text": "Robotics Lab",
            "needs_clarification": False,
            "confidence": 0.88,
        }).to_intent_result(raw_query="فين مختبر الروبوتات؟")

        assert isinstance(result, IntentResult)
        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.language == "ar-EG"
        assert result.target_text == "Robotics Lab"
        assert result.raw_query == "فين مختبر الروبوتات؟"
        assert result.confidence == 0.88

    def test_to_intent_result_navigation(self) -> None:
        result = RouterRawOutput.model_validate({
            "intent": "navigation_request",
            "target_text": "Lab 214",
        }).to_intent_result()

        assert result.intent == IntentClass.NAVIGATION_REQUEST
        assert result.target_text == "Lab 214"

    def test_defaults_are_safe(self) -> None:
        """Minimum valid input — only intent required."""
        model = RouterRawOutput.model_validate({"intent": "social_chat"})
        assert model.language == "en"
        assert model.needs_clarification is False
        assert model.target_text is None
        assert model.confidence == 0.0


class TestParseRouterResponse:
    def test_valid_json_returns_intent_result(self) -> None:
        raw = json.dumps({"intent": "social_chat", "language": "en", "confidence": 0.9})
        result = parse_router_response(raw, raw_query="how are you")
        assert result is not None
        assert result.intent == IntentClass.SOCIAL_CHAT

    def test_malformed_json_returns_none(self) -> None:
        assert parse_router_response("not json at all") is None

    def test_invalid_intent_returns_none(self) -> None:
        raw = json.dumps({"intent": "random_garbage"})
        assert parse_router_response(raw) is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_router_response("") is None


# ─────────────────────────────────────────────────────────────────────────────
# TestCompleteJson — low-level retry behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCompleteJson:
    def test_successful_response_returned_as_string(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({"intent": "campus_query", "language": "en"})
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.complete_json(SYSTEM_PROMPT, TRANSCRIPT)

        assert result == payload
        groq_client._client.chat.completions.create.assert_called_once()

    def test_timeout_retries_all_attempts_then_returns_none(
        self, groq_client: GroqClient
    ) -> None:
        groq_client._client.chat.completions.create.side_effect = _make_timeout_error()

        result = groq_client.complete_json(SYSTEM_PROMPT, TRANSCRIPT)

        assert result is None
        assert groq_client._client.chat.completions.create.call_count == 3

    def test_retry_success_on_second_attempt(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({"intent": "social_chat"})
        groq_client._client.chat.completions.create.side_effect = [
            _make_timeout_error(),
            _make_groq_response(payload),
        ]

        result = groq_client.complete_json(SYSTEM_PROMPT, TRANSCRIPT)

        assert result == payload
        assert groq_client._client.chat.completions.create.call_count == 2

    def test_unexpected_exception_stops_immediately_without_retrying(
        self, groq_client: GroqClient
    ) -> None:
        """Non-retriable errors must not be retried."""
        groq_client._client.chat.completions.create.side_effect = RuntimeError("sdk crash")

        result = groq_client.complete_json(SYSTEM_PROMPT, TRANSCRIPT)

        assert result is None
        assert groq_client._client.chat.completions.create.call_count == 1

    def test_connection_error_retries(self, groq_client: GroqClient) -> None:
        groq_client._client.chat.completions.create.side_effect = _make_connection_error()

        result = groq_client.complete_json(SYSTEM_PROMPT, TRANSCRIPT)

        assert result is None
        assert groq_client._client.chat.completions.create.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestCallRouter — high-level intent classification
# ─────────────────────────────────────────────────────────────────────────────

class TestCallRouter:
    def test_campus_query_classified_correctly(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({
            "intent": "campus_query",
            "language": "en",
            "target_text": "robotics lab",
            "needs_clarification": False,
            "confidence": 0.95,
            "reason": "user asked about location",
        })
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.call_router(SYSTEM_PROMPT, TRANSCRIPT)

        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.target_text == "robotics lab"
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.raw_query == TRANSCRIPT

    def test_navigation_request_classified_correctly(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({
            "intent": "navigation_request",
            "language": "en",
            "target_text": "Lab 214",
            "needs_clarification": False,
            "confidence": 0.97,
        })
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.call_router(SYSTEM_PROMPT, "take me to lab 214")

        assert result.intent == IntentClass.NAVIGATION_REQUEST
        assert result.target_text == "Lab 214"

    def test_social_chat_classified_correctly(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({"intent": "social_chat", "language": "en", "confidence": 0.91})
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.call_router(SYSTEM_PROMPT, "how are you?")

        assert result.intent == IntentClass.SOCIAL_CHAT

    def test_malformed_json_returns_unknown(
        self, groq_client: GroqClient
    ) -> None:
        groq_client._client.chat.completions.create.return_value = _make_groq_response(
            "this is not json at all"
        )

        result = groq_client.call_router(SYSTEM_PROMPT, TRANSCRIPT)

        assert result.intent == IntentClass.UNKNOWN
        assert result.raw_query == TRANSCRIPT
        assert result.reason == "json_parse_error"

    def test_invalid_intent_in_json_returns_unknown(
        self, groq_client: GroqClient
    ) -> None:
        payload = json.dumps({"intent": "launch_rockets", "language": "en"})
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.call_router(SYSTEM_PROMPT, TRANSCRIPT)

        assert result.intent == IntentClass.UNKNOWN
        assert result.reason == "validation_error"

    def test_timeout_returns_unknown(self, groq_client: GroqClient) -> None:
        groq_client._client.chat.completions.create.side_effect = _make_timeout_error()

        result = groq_client.call_router(SYSTEM_PROMPT, TRANSCRIPT)

        assert result.intent == IntentClass.UNKNOWN
        assert result.reason == "no_response"

    def test_retry_success_after_one_timeout(
        self, groq_client: GroqClient
    ) -> None:
        """Router must succeed and return correct intent when retry recovers."""
        payload = json.dumps({"intent": "social_chat", "language": "en", "confidence": 0.88})
        groq_client._client.chat.completions.create.side_effect = [
            _make_timeout_error(),
            _make_groq_response(payload),
        ]

        result = groq_client.call_router(SYSTEM_PROMPT, "how are you?")

        assert result.intent == IntentClass.SOCIAL_CHAT
        assert groq_client._client.chat.completions.create.call_count == 2

    def test_all_retries_exhausted_returns_unknown(
        self, groq_client: GroqClient
    ) -> None:
        """After all retries fail, call_router must still return a safe result."""
        groq_client._client.chat.completions.create.side_effect = _make_timeout_error()

        result = groq_client.call_router(SYSTEM_PROMPT, TRANSCRIPT)

        assert result.intent == IntentClass.UNKNOWN
        assert groq_client._client.chat.completions.create.call_count == 3

    def test_arabic_transcript_preserved_in_raw_query(
        self, groq_client: GroqClient
    ) -> None:
        """Arabic transcript text must survive through to IntentResult.raw_query."""
        arabic = "فين مختبر الروبوتات؟"
        payload = json.dumps({
            "intent": "campus_query",
            "language": "ar-EG",
            "target_text": "Robotics Lab",
            "confidence": 0.89,
        })
        groq_client._client.chat.completions.create.return_value = _make_groq_response(payload)

        result = groq_client.call_router(SYSTEM_PROMPT, arabic)

        assert result.intent == IntentClass.CAMPUS_QUERY
        assert result.language == "ar-EG"
        assert result.raw_query == arabic


# ─────────────────────────────────────────────────────────────────────────────
# TestUnknownResult helper
# ─────────────────────────────────────────────────────────────────────────────

class TestUnknownResult:
    def test_defaults(self) -> None:
        r = _unknown_result()
        assert r.intent == IntentClass.UNKNOWN
        assert r.language == "en"
        assert r.raw_query is None

    def test_with_query_and_reason(self) -> None:
        r = _unknown_result(raw_query="some text", reason="json_parse_error")
        assert r.raw_query == "some text"
        assert r.reason == "json_parse_error"
