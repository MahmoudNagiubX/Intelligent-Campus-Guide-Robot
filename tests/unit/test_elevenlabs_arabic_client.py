"""
Unit tests for ElevenLabsArabicClient.

No network calls are made; live websocket behavior is covered by parsing and
mock-mode tests only.
"""

from __future__ import annotations

import json

import pytest

from app.stt.elevenlabs_arabic_client import (
    ElevenLabsArabicClient,
    _is_arabic_lang,
    _normalise_lang,
)


def _make_client(**kwargs) -> ElevenLabsArabicClient:
    return ElevenLabsArabicClient(mock=True, **kwargs)


def _ws_message(
    text: str,
    msg_type: str = "final",
    lang: str = "ar-EG",
    confidence: float = 0.95,
) -> str:
    return json.dumps(
        {
            "type": "transcript",
            "transcript": {
                "text": text,
                "type": msg_type,
                "language_code": lang,
                "confidence": confidence,
            },
        }
    )


def test_has_required_interface_methods():
    client = _make_client()

    assert callable(client.connect)
    assert callable(client.disconnect)
    assert callable(client.send_audio)
    assert callable(client.finalize_turn)
    assert callable(client.inject_mock_transcript)
    assert callable(client.set_callbacks)
    assert callable(client.set_session_id)
    assert callable(client.reset_turn)


def test_connect_sets_connected_flag():
    client = _make_client()

    client.connect()

    assert client._connected is True


def test_disconnect_clears_connected_flag():
    client = _make_client()
    client.connect()

    client.disconnect()

    assert client._connected is False


def test_inject_mock_arabic_final_fires_on_final():
    finals = []
    client = _make_client(on_final=finals.append)
    client.connect()

    client.inject_mock_transcript("أين المختبر", language="ar-EG", language_confidence=0.96)

    assert len(finals) == 1
    assert finals[0].text == "أين المختبر"
    assert finals[0].language == "ar-EG"
    assert finals[0].language_confidence == 0.96
    assert finals[0].source == "elevenlabs_mock"


def test_inject_mock_partial_fires_on_partial():
    partials = []
    client = _make_client(on_partial=partials.append)
    client.connect()

    client.inject_mock_transcript("أين", is_final=False, language="ar-EG")

    assert len(partials) == 1
    assert partials[0].is_final is False


def test_inject_mock_in_real_mode_is_ignored():
    finals = []
    client = ElevenLabsArabicClient(mock=False, on_final=finals.append)

    client.inject_mock_transcript("test")

    assert finals == []


def test_non_arabic_result_is_dropped():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("where is the lab", lang="en"))

    assert finals == []


def test_unknown_lang_result_is_dropped():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("some text", lang="unknown"))

    assert finals == []


def test_arabic_result_reaches_on_final():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("أين مكتب الدكتور", lang="ar-EG"))

    assert len(finals) == 1
    assert finals[0].language == "ar-EG"
    assert finals[0].source == "elevenlabs"


def test_arabic_result_with_ar_only_code_reaches_on_final():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("فين المختبر", lang="ar"))

    assert len(finals) == 1
    assert finals[0].language == "ar"


def test_duplicate_final_is_not_emitted_twice():
    finals = []
    client = _make_client(on_final=finals.append)
    text = "أين المختبر"

    client._parse_and_dispatch(_ws_message(text, lang="ar-EG"))
    client._parse_and_dispatch(_ws_message(text, lang="ar-EG"))

    assert len(finals) == 1


def test_different_text_after_dedup_is_emitted():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("أين المختبر", lang="ar-EG"))
    client._parse_and_dispatch(_ws_message("فين الدكتور", lang="ar-EG"))

    assert len(finals) == 2


def test_reset_turn_clears_dedup_guard():
    finals = []
    client = _make_client(on_final=finals.append)
    text = "أين المختبر"

    client._parse_and_dispatch(_ws_message(text, lang="ar-EG"))
    client.reset_turn()
    client._parse_and_dispatch(_ws_message(text, lang="ar-EG"))

    assert len(finals) == 2


def test_partial_arabic_reaches_on_partial():
    partials = []
    client = _make_client(on_partial=partials.append)

    client._parse_and_dispatch(_ws_message("أين", msg_type="partial", lang="ar-EG"))

    assert len(partials) == 1
    assert partials[0].is_final is False


def test_partial_english_is_dropped():
    partials = []
    client = _make_client(on_partial=partials.append)

    client._parse_and_dispatch(_ws_message("where", msg_type="partial", lang="en"))

    assert partials == []


def test_set_callbacks_replaces_on_final():
    first = []
    second = []
    client = _make_client(on_final=first.append)

    client.set_callbacks(on_final=second.append)
    client.inject_mock_transcript("أين", language="ar-EG")

    assert first == []
    assert len(second) == 1


def test_set_session_id_propagates_to_events():
    finals = []
    client = _make_client(on_final=finals.append)

    client.set_session_id("session-abc-123")
    client.inject_mock_transcript("أين المختبر", language="ar-EG")

    assert finals[0].session_id == "session-abc-123"


def test_api_error_message_fires_on_error():
    errors = []
    client = _make_client()
    client.set_callbacks(on_error=lambda code, message: errors.append((code, message)))

    client._parse_and_dispatch(
        json.dumps(
            {
                "type": "error",
                "error": {"code": 4001, "message": "invalid key"},
            }
        )
    )

    assert len(errors) == 1
    assert errors[0][0] == "elevenlabs_api_error"


def test_malformed_json_does_not_raise():
    client = _make_client()

    client._parse_and_dispatch("{bad json :::}")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("ar", True),
        ("ar-EG", True),
        ("ar-SA", True),
        ("AR-EG", True),
        ("en", False),
        ("en-US", False),
        ("unknown", False),
        ("", False),
    ],
)
def test_is_arabic_lang(code, expected):
    assert _is_arabic_lang(code) is expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("ar-EG", "ar-EG"),
        ("ar-eg", "ar-EG"),
        ("ar", "ar"),
        ("ar-SA", "ar"),
        ("en", "en"),
    ],
)
def test_normalise_lang(code, expected):
    assert _normalise_lang(code) == expected


def test_real_transcripts_have_elevenlabs_source():
    finals = []
    client = _make_client(on_final=finals.append)

    client._parse_and_dispatch(_ws_message("أين المختبر", lang="ar-EG"))

    assert finals[0].source == "elevenlabs"


def test_mock_transcripts_have_elevenlabs_mock_source():
    finals = []
    client = _make_client(on_final=finals.append)

    client.inject_mock_transcript("أين المختبر", language="ar-EG")

    assert finals[0].source == "elevenlabs_mock"


def test_build_ws_url_uses_settings_and_keyterm_limit(monkeypatch):
    from app.config.settings import get_settings

    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_MODEL", "scribe_v2")
    monkeypatch.setenv("ELEVENLABS_KEYTERMS_MAX", "2")
    get_settings.cache_clear()
    try:
        client = ElevenLabsArabicClient(mock=False, keyterms=["lab", "office", "third"])
        url = client._build_ws_url()
        assert "model_id=scribe_v2" in url
        assert "xi-api-key=test-key" in url
        assert "keywords=lab%2Coffice" in url
        assert "third" not in url
    finally:
        get_settings.cache_clear()
