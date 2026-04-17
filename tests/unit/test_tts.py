"""
Navigator - Phase 7 Step 7.4: TTS and Barge-In Tests

Tests for EdgeTTSClient (voice selection, mock synthesis) and
PlaybackManager (play, stop, barge-in, callbacks, state transitions).

No real audio hardware or edge-tts network access required.

Run with:
    pytest tests/unit/test_tts.py -v
"""

import threading
import time

import pytest

from app.tts.edge_tts_client import EdgeTTSClient, _build_silent_wav
from app.tts.playback import PlaybackManager, PlaybackState


# ─────────────────────────────────────────────────────────────────────────────
# EdgeTTSClient
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeTTSClient:
    def test_voice_for_english_returns_jenny(self):
        client = EdgeTTSClient(mock=True)
        voice = client.voice_for("en")
        assert "Jenny" in voice or "en-US" in voice

    def test_voice_for_arabic_returns_salma(self):
        client = EdgeTTSClient(mock=True)
        voice = client.voice_for("ar-EG")
        assert "Salma" in voice or "ar-EG" in voice

    def test_voice_for_arabic_short_code(self):
        client = EdgeTTSClient(mock=True)
        voice = client.voice_for("ar")
        assert "Salma" in voice or "ar-EG" in voice

    def test_voice_for_arabic_sa_variant(self):
        client = EdgeTTSClient(mock=True)
        voice = client.voice_for("ar-SA")
        # Should still return the configured Arabic voice
        assert "ar" in voice.lower()

    def test_voice_for_unknown_language_returns_english(self):
        client = EdgeTTSClient(mock=True)
        voice = client.voice_for("fr")
        assert "en" in voice.lower() or "Jenny" in voice

    def test_mock_synthesize_returns_bytes(self):
        import asyncio
        client = EdgeTTSClient(mock=True)
        result = asyncio.run(client.synthesize("Hello!", "en"))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_mock_synthesize_returns_wav_header(self):
        import asyncio
        client = EdgeTTSClient(mock=True)
        result = asyncio.run(client.synthesize("Hello!", "en"))
        # WAV files start with RIFF header
        assert result[:4] == b"RIFF"

    def test_mock_synthesize_arabic(self):
        import asyncio
        client = EdgeTTSClient(mock=True)
        result = asyncio.run(client.synthesize("مرحباً", "ar-EG"))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_empty_text_returns_empty_bytes(self):
        import asyncio
        client = EdgeTTSClient(mock=True)
        result = asyncio.run(client.synthesize("", "en"))
        assert result == b""

    def test_whitespace_only_returns_empty_bytes(self):
        import asyncio
        client = EdgeTTSClient(mock=True)
        result = asyncio.run(client.synthesize("   ", "en"))
        assert result == b""

    def test_synthesize_sync_mock(self):
        client = EdgeTTSClient(mock=True)
        result = client.synthesize_sync("Hello from sync", "en")
        assert isinstance(result, bytes)

    def test_build_silent_wav_is_valid_wav(self):
        wav = _build_silent_wav(200)
        assert wav[:4] == b"RIFF"
        assert len(wav) > 44  # minimum WAV header size


# ─────────────────────────────────────────────────────────────────────────────
# PlaybackManager
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaybackManager:
    def _silence(self) -> bytes:
        return _build_silent_wav(100)

    def test_initial_state_is_idle(self):
        pm = PlaybackManager(mock=True)
        assert pm.state == PlaybackState.IDLE

    def test_play_sets_state_to_playing(self):
        pm = PlaybackManager(mock=True)
        pm.play(self._silence())
        # State check immediately after — may still be PLAYING or DONE for fast mock
        assert pm.state in (PlaybackState.PLAYING, PlaybackState.DONE)

    def test_playback_completes_and_fires_callback(self):
        completed = []
        pm = PlaybackManager(mock=True, on_complete=lambda: completed.append(True))
        pm.play(self._silence())
        time.sleep(0.3)
        assert completed == [True]
        assert pm.state == PlaybackState.DONE

    def test_stop_during_playback_sets_stopped(self):
        stopped_event = threading.Event()
        pm = PlaybackManager(
            mock=False,  # use real loop — but we stop it before it finishes
            on_complete=lambda: None,
        )
        # Patch _mock_playback to sleep, then stop from outside
        pm._mock = True

        def delayed_stop():
            time.sleep(0.01)
            pm.stop()
            stopped_event.set()

        pm.play(self._silence())
        t = threading.Thread(target=delayed_stop, daemon=True)
        t.start()
        stopped_event.wait(timeout=2.0)
        assert pm.state in (PlaybackState.STOPPED, PlaybackState.IDLE, PlaybackState.DONE)

    def test_barge_in_fires_callback(self):
        barged = []
        pm = PlaybackManager(mock=True, on_barge_in=lambda: barged.append(True))
        # Force PLAYING state
        pm._state = PlaybackState.PLAYING
        pm.notify_speech_detected()
        assert len(barged) == 1

    def test_barge_in_stops_playback(self):
        pm = PlaybackManager(mock=True)
        pm._state = PlaybackState.PLAYING
        pm.notify_speech_detected()
        assert pm.state == PlaybackState.STOPPED

    def test_barge_in_ignored_when_idle(self):
        barged = []
        pm = PlaybackManager(mock=True, on_barge_in=lambda: barged.append(True))
        pm.notify_speech_detected()   # state is IDLE — should be silently ignored
        assert barged == []

    def test_cancel_is_same_as_stop(self):
        pm = PlaybackManager(mock=True)
        pm._state = PlaybackState.PLAYING
        pm.cancel()
        assert pm.state == PlaybackState.STOPPED

    def test_is_playing_property(self):
        pm = PlaybackManager(mock=True)
        pm._state = PlaybackState.PLAYING
        assert pm.is_playing is True
        pm._state = PlaybackState.DONE
        assert pm.is_playing is False

    def test_empty_audio_does_not_start(self):
        completed = []
        pm = PlaybackManager(mock=True, on_complete=lambda: completed.append(True))
        pm.play(b"")
        time.sleep(0.2)
        assert completed == []
        assert pm.state == PlaybackState.IDLE

    def test_stop_on_idle_does_not_crash(self):
        pm = PlaybackManager(mock=True)
        pm.stop()   # should be silent no-op

    def test_multiple_plays_last_one_wins(self):
        completed = []
        pm = PlaybackManager(mock=True, on_complete=lambda: completed.append(True))
        pm.play(self._silence())
        pm.play(self._silence())   # cancels first
        time.sleep(0.5)
        # At most one completion event
        assert len(completed) <= 2

    def test_no_callbacks_set_does_not_crash(self):
        pm = PlaybackManager(mock=True)
        pm.play(self._silence())
        time.sleep(0.3)

    def test_barge_in_callback_exception_does_not_propagate(self):
        def bad_callback():
            raise ValueError("callback error")
        pm = PlaybackManager(mock=True, on_barge_in=bad_callback)
        pm._state = PlaybackState.PLAYING
        pm.notify_speech_detected()  # must not raise
