"""Tests that social answers respect the detected language."""

from unittest.mock import MagicMock

from app.pipeline.response_composer import ResponseComposer, _contains_arabic_script


def _make_composer(mock_response: str) -> ResponseComposer:
    groq = MagicMock()
    groq.complete_text.return_value = mock_response
    return ResponseComposer(groq=groq)


def test_english_input_english_output():
    composer = _make_composer("Happy to help!")
    packet = composer.compose_social_answer("How are you?", language="en")

    assert not _contains_arabic_script(packet.text)
    assert packet.language == "en"


def test_english_input_llm_returns_arabic_fallback():
    """If the LLM returns Arabic for English input, use English fallback."""
    composer = _make_composer("أهلاً! أقدر أساعدك؟")
    packet = composer.compose_social_answer("How are you?", language="en")

    assert not _contains_arabic_script(packet.text)
    assert packet.language == "en"


def test_arabic_input_arabic_output():
    composer = _make_composer("أهلاً! كيف أساعدك؟")
    packet = composer.compose_social_answer("مرحباً", language="ar")

    assert _contains_arabic_script(packet.text)
    assert packet.language == "ar"


def test_contains_arabic_script():
    assert _contains_arabic_script("أهلاً") is True
    assert _contains_arabic_script("Hello") is False
    assert _contains_arabic_script("room 214") is False
    assert _contains_arabic_script("room أوضة 214") is True
