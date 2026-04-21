from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from app.config.settings import get_settings
from app.stt.deepgram_client import DeepgramStreamingClient, load_keyterms_from_db


def _results_message(
    text: str,
    *,
    is_final: bool,
    speech_final: bool = False,
    language: str = "en",
    language_confidence: float | None = None,
    confidence: float = 0.95,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="Results",
        channel=SimpleNamespace(
            alternatives=[
                SimpleNamespace(
                    transcript=text,
                    confidence=confidence,
                    language=language,
                    language_confidence=language_confidence,
                )
            ]
        ),
        is_final=is_final,
        speech_final=speech_final,
        language=language,
        language_confidence=language_confidence,
    )


def test_mock_injection_forwards_language_and_confidence() -> None:
    finals = []
    client = DeepgramStreamingClient(mock=True, on_final=lambda event: finals.append(event))
    client.connect()
    client.inject_mock_transcript("فين معمل الروبوتات", language="ar-EG", language_confidence=0.96)

    assert finals[0].language == "ar-EG"
    assert finals[0].language_confidence == 0.96


def test_partial_transcript_forwards_detected_language_metadata() -> None:
    partials = []
    client = DeepgramStreamingClient(mock=True, on_partial=lambda event: partials.append(event))

    client._handle_deepgram_message(
        _results_message("where is", is_final=False, language="en", language_confidence=0.88)
    )

    assert partials[0].language == "en"
    assert partials[0].language_confidence == 0.88


def test_final_transcript_flush_forwards_detected_language_metadata() -> None:
    finals = []
    client = DeepgramStreamingClient(mock=True, on_final=lambda event: finals.append(event))

    client._handle_deepgram_message(
        _results_message(
            "معمل الروبوتات",
            is_final=True,
            speech_final=True,
            language="ar-EG",
            language_confidence=0.97,
        )
    )

    assert finals[0].language == "ar-EG"
    assert finals[0].language_confidence == 0.97


def test_build_connect_options_uses_multi_language_for_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_LANGUAGE", "multi")
    get_settings.cache_clear()
    try:
        client = DeepgramStreamingClient(mock=True, language="en", keyterms=["Robotics Lab"])
        options = client._build_connect_options()
    finally:
        get_settings.cache_clear()

    assert options["language"] == "multi"
    assert options["model"] == "nova-3"
    assert "keyterm" not in options


def test_build_connect_options_forces_en_when_keyterm_payload_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPGRAM_LANGUAGE", "multi")
    get_settings.cache_clear()
    try:
        client = DeepgramStreamingClient(mock=True, language="ar-EG", keyterms=["Robotics Lab"])
        monkeypatch.setattr(client, "_build_nova3_keyterm_options", lambda: {"keyterm": ["Robotics Lab"]})
        options = client._build_connect_options()
    finally:
        get_settings.cache_clear()

    assert options["language"] == "en"
    assert options["keyterm"] == ["Robotics Lab"]


def test_load_keyterms_reads_new_bilingual_truth_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE rooms (id INTEGER PRIMARY KEY, room_name TEXT, is_active INTEGER, lang TEXT);
        CREATE TABLE labs (id INTEGER PRIMARY KEY, lab_name TEXT, is_active INTEGER, lang TEXT);
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, is_active INTEGER, lang TEXT);
        CREATE TABLE landmarks (id INTEGER PRIMARY KEY, landmark_name TEXT, is_active INTEGER, lang TEXT);
        CREATE TABLE staff (id INTEGER PRIMARY KEY, full_name TEXT, is_active INTEGER, lang TEXT);
        CREATE TABLE aliases (
            id INTEGER PRIMARY KEY,
            canonical_type TEXT,
            canonical_id INTEGER,
            alias_text TEXT,
            lang TEXT
        );
        INSERT INTO rooms VALUES (1, 'Robotics Lab', 1, 'en');
        INSERT INTO labs VALUES (1, 'Robotics and Machine Vision', 1, 'en');
        INSERT INTO departments VALUES (1, 'Software Engineering Department', 1, 'en');
        INSERT INTO landmarks VALUES (1, 'Main Library', 1, 'en');
        INSERT INTO staff VALUES (1, 'Dr. Sara Ali', 1, 'en');
        INSERT INTO aliases VALUES (1, 'room', 1, 'robot room', 'en');
        """
    )
    monkeypatch.setattr("app.stt.deepgram_client.get_db", lambda: conn)

    terms = load_keyterms_from_db()

    assert "Robotics Lab" in terms
    assert "Robotics and Machine Vision" in terms
    assert "Software Engineering Department" in terms
    assert "Main Library" in terms
    assert "Dr. Sara Ali" in terms
    assert "robot room" in terms
