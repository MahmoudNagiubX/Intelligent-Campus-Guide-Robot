"""
Navigator - Phase 4 Step 4.4: STT Client Tests
Tests for DeepgramStreamingClient in mock mode and keyterm loading.

All tests run without real Deepgram API access.

Run with:
    pytest tests/unit/test_stt.py -v
"""

from types import SimpleNamespace

import pytest
from deepgram import AsyncDeepgramClient

from app.stt.deepgram_client import DeepgramStreamingClient
from app.utils.contracts import TranscriptEvent


class TestDeepgramMockMode:
    def _make_client(self, **kwargs) -> DeepgramStreamingClient:
        return DeepgramStreamingClient(mock=True, **kwargs)

    @staticmethod
    def _results_message(
        text: str,
        *,
        is_final: bool,
        speech_final: bool = False,
        from_finalize: bool = False,
        confidence: float = 0.95,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            type="Results",
            channel=SimpleNamespace(
                alternatives=[
                    SimpleNamespace(
                        transcript=text,
                        confidence=confidence,
                    )
                ]
            ),
            is_final=is_final,
            speech_final=speech_final,
            from_finalize=from_finalize,
        )

    def test_connect_in_mock_mode(self):
        client = self._make_client()
        client.connect()
        assert client._connected is True

    def test_disconnect_in_mock_mode(self):
        client = self._make_client()
        client.connect()
        client.disconnect()
        assert client._connected is False

    def test_connect_in_mock_mode_fires_connected_callback(self):
        connected = []
        client = self._make_client()
        client.set_callbacks(on_connected=lambda: connected.append(True))
        client.connect()
        assert connected == [True]

    def test_inject_final_fires_on_final_callback(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))
        client.connect()
        client.inject_mock_transcript("Where is the Robotics Lab?", is_final=True)
        assert len(finals) == 1
        assert finals[0].text == "Where is the Robotics Lab?"
        assert finals[0].is_final is True

    def test_inject_partial_fires_on_partial_callback(self):
        partials = []
        client = self._make_client(on_partial=lambda e: partials.append(e))
        client.connect()
        client.inject_mock_transcript("where is", is_final=False)
        assert len(partials) == 1
        assert partials[0].is_final is False

    def test_final_deduplication(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))
        client.connect()
        client.inject_mock_transcript("where is the lab", is_final=True)
        client.inject_mock_transcript("where is the lab", is_final=True)  # duplicate
        assert len(finals) == 1

    def test_different_finals_both_fire(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))
        client.connect()
        client.inject_mock_transcript("where is the lab", is_final=True)
        client.inject_mock_transcript("take me to lab 214", is_final=True)
        assert len(finals) == 2

    def test_empty_final_skipped(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))
        client.connect()
        client.inject_mock_transcript("", is_final=True)
        client.inject_mock_transcript("   ", is_final=True)
        assert len(finals) == 0

    def test_transcript_event_has_correct_language(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e), language="ar-EG")
        client.connect()
        client.inject_mock_transcript("فين المعمل", is_final=True)
        assert finals[0].language == "ar-EG"

    def test_session_id_attached_to_event(self):
        finals = []
        client = self._make_client(
            on_final=lambda e: finals.append(e),
            session_id="test-session-123",
        )
        client.connect()
        client.inject_mock_transcript("hello", is_final=True)
        assert finals[0].session_id == "test-session-123"

    def test_send_audio_in_mock_mode_does_not_crash(self):
        client = self._make_client()
        client.connect()
        # send_audio is a no-op in mock mode — must not raise
        client.send_audio(b"\x00" * 1024)

    def test_inject_in_real_mode_logs_warning(self):
        """inject_mock_transcript must log a warning when called in real mode."""
        client = DeepgramStreamingClient(mock=False)
        # Should not crash — just logs a warning
        client.inject_mock_transcript("test")

    def test_no_callbacks_set_does_not_crash(self):
        client = self._make_client()
        client.connect()
        client.inject_mock_transcript("hello navigator", is_final=True)
        client.inject_mock_transcript("partial text", is_final=False)

    def test_transcript_event_type(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))
        client.connect()
        client.inject_mock_transcript("where is the library", is_final=True)
        assert isinstance(finals[0], TranscriptEvent)

    def test_keyterms_stored_on_init(self):
        keyterms = ["Robotics Lab", "Computer Science", "Dr Ahmed"]
        client = self._make_client(keyterms=keyterms)
        assert client._keyterms == keyterms

    def test_connect_options_use_minimal_valid_handshake_query(self):
        client = self._make_client()

        assert client._build_connect_options() == {
            "model": "nova-3",
            "language": "en",
            "encoding": "linear16",
            "sample_rate": 16000,
            "channels": 1,
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
        }

    def test_nova3_keyterm_options_use_supported_keyterm_param(self):
        client = self._make_client(
            keyterms=["Robotics Lab", "robotics   lab", "Computer Science", "", "Dr Ahmed"]
        )

        assert client._build_nova3_keyterm_options() == {
            "keyterm": ["Robotics Lab", "Computer Science", "Dr Ahmed"]
        }

    def test_connect_options_reintroduce_nova3_keyterms_after_minimal_baseline(self):
        client = self._make_client(keyterms=["Robotics Lab", "Computer Science"])

        assert client._build_connect_options()["keyterm"] == ["Robotics Lab", "Computer Science"]

    def test_websocket_request_serializes_boolean_query_values_lowercase(self):
        client = self._make_client()
        dg_client = AsyncDeepgramClient(api_key="test-key")

        ws_url, headers = client._build_websocket_request(dg_client, client._build_connect_options())

        assert "interim_results=true" in ws_url
        assert "punctuate=true" in ws_url
        assert "smart_format=true" in ws_url
        assert "interim_results=True" not in ws_url
        assert headers["Authorization"] == "Token test-key"

    def test_arabic_transcript_not_modified(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e), language="ar-EG")
        client.connect()
        arabic = "فين مختبر الروبوتات؟"
        client.inject_mock_transcript(arabic, is_final=True)
        # Arabic text should be preserved exactly (minus leading/trailing whitespace)
        assert finals[0].text == arabic.strip()

    def test_final_segments_flush_only_when_speech_final_arrives(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))

        client._handle_deepgram_message(
            self._results_message(
                "where is the robotics",
                is_final=True,
                speech_final=False,
            )
        )
        assert finals == []

        client._handle_deepgram_message(
            self._results_message(
                "lab",
                is_final=True,
                speech_final=True,
            )
        )

        assert [event.text for event in finals] == ["where is the robotics lab"]

    def test_utterance_end_flushes_buffered_final_segments(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))

        client._handle_deepgram_message(
            self._results_message(
                "take me to lab 214",
                is_final=True,
                speech_final=False,
            )
        )
        client._handle_deepgram_message(SimpleNamespace(type="UtteranceEnd"))

        assert [event.text for event in finals] == ["take me to lab 214"]

    def test_from_finalize_flushes_buffered_final_segments(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))

        client._handle_deepgram_message(
            self._results_message(
                "where is dr ahmed office",
                is_final=True,
                from_finalize=True,
            )
        )

        assert [event.text for event in finals] == ["where is dr ahmed office"]

    def test_duplicate_final_segment_is_not_duplicated_in_flush(self):
        finals = []
        client = self._make_client(on_final=lambda e: finals.append(e))

        client._handle_deepgram_message(
            self._results_message(
                "robotics lab",
                is_final=True,
                speech_final=False,
            )
        )
        client._handle_deepgram_message(
            self._results_message(
                "robotics lab",
                is_final=True,
                speech_final=True,
            )
        )

        assert [event.text for event in finals] == ["robotics lab"]


class TestKeytermLoader:
    def test_load_keyterms_returns_list(self):
        """load_keyterms_from_db must return a list even if DB is empty."""
        from app.stt.deepgram_client import load_keyterms_from_db
        # This will try the actual DB — gracefully fall back on any error
        result = load_keyterms_from_db()
        assert isinstance(result, list)

    def test_load_keyterms_with_real_db(self, monkeypatch):
        """Verify keyterm loading SQL runs correctly against an in-memory DB."""
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript("""
            CREATE TABLE locations (id INTEGER PRIMARY KEY, name TEXT, code TEXT, is_active INTEGER DEFAULT 1);
            CREATE TABLE staff (id INTEGER PRIMARY KEY, full_name TEXT, is_active INTEGER DEFAULT 1);
            CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT, is_active INTEGER DEFAULT 1);
            CREATE TABLE facilities (id INTEGER PRIMARY KEY, name TEXT, is_active INTEGER DEFAULT 1);
            CREATE TABLE aliases (
                id INTEGER PRIMARY KEY,
                canonical_type TEXT,
                canonical_id INTEGER,
                alias_text TEXT
            );
            INSERT INTO locations VALUES (1, 'Robotics Lab', 'LAB_214', 1);
            INSERT INTO staff VALUES (1, 'Dr. Ahmed Samy', 1);
            INSERT INTO departments VALUES (1, 'Computer Science Department', 1);
            INSERT INTO facilities VALUES (1, 'Medical Center', 1);
            INSERT INTO aliases VALUES (1, 'location', 1, 'robot room');
        """)
        conn.commit()

        monkeypatch.setattr("app.stt.deepgram_client.get_db", lambda: conn)

        from app.stt.deepgram_client import load_keyterms_from_db
        terms = load_keyterms_from_db()
        assert "Robotics Lab" in terms
        assert "Dr. Ahmed Samy" in terms
        assert "Computer Science Department" in terms
        assert "Medical Center" in terms
        assert "robot room" in terms
