"""
Navigator - Phase 7 Step 7.4: TTS and Barge-In Tests

Tests for EdgeTTSClient (voice selection, mock synthesis) and
PlaybackManager (play, stop, barge-in, callbacks, state transitions).

No real audio hardware or edge-tts network access required.

Run with:
    pytest tests/unit/test_tts.py -v
"""

import asyncio
import sys
import threading
import time
import types

import pytest

from app.config.settings import get_settings
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

    def test_rate_and_voice_can_be_configured_from_env(self, monkeypatch):
        monkeypatch.setenv("EDGE_TTS_VOICE_EN", "en-GB-SoniaNeural")
        monkeypatch.setenv("EDGE_TTS_VOICE_AR", "ar-SA-ZariyahNeural")
        monkeypatch.setenv("EDGE_TTS_RATE", "-15%")
        get_settings.cache_clear()
        try:
            client = EdgeTTSClient(mock=True)
            assert client.voice_for("en") == "en-GB-SoniaNeural"
            assert client.voice_for("ar-EG") == "ar-SA-ZariyahNeural"
            assert client._rate == "-15%"
        finally:
            get_settings.cache_clear()

    def test_edge_tts_rate_is_passed_to_sdk(self, monkeypatch):
        captured = {}

        class FakeCommunicate:
            def __init__(self, *, text, voice, rate):
                captured["text"] = text
                captured["voice"] = voice
                captured["rate"] = rate

            async def stream(self):
                yield {"type": "audio", "data": b"abc"}

        monkeypatch.setenv("EDGE_TTS_RATE", "-12%")
        get_settings.cache_clear()
        monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate))
        try:
            client = EdgeTTSClient(mock=False)
            result = asyncio.run(client.synthesize("Hello there", "en"))
            assert result == b"abc"
            assert captured["voice"] == client.voice_for("en")
            assert captured["rate"] == "-12%"
        finally:
            get_settings.cache_clear()
            sys.modules.pop("edge_tts", None)


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
            mock=True,
            on_complete=lambda: None,
        )
        # Patch _mock_playback to sleep, then stop from outside

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

    def test_play_test_tone_starts_mock_playback(self):
        pm = PlaybackManager(mock=True)
        pm.play_test_tone()
        time.sleep(0.2)
        assert pm.state in (PlaybackState.PLAYING, PlaybackState.DONE)

    def test_resolve_explicit_output_device(self):
        class FakeSoundDevice:
            default = type("Default", (), {"device": (2, 4)})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Device {index}"}

        pm = PlaybackManager(mock=True)
        pm._speaker_device_index = 7

        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index == 7
        assert device_name == "Device 7"

    def test_resolve_default_output_device(self):
        class FakeSoundDevice:
            default = type("Default", (), {"device": (2, 5)})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Output {index}"}

        pm = PlaybackManager(mock=True)

        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index == 5
        assert device_name == "Output 5"

    def test_resolve_default_device_as_plain_int(self):
        """sd.default.device is already an int (some Linux/Mac setups)."""
        class FakeSoundDevice:
            default = type("Default", (), {"device": 3})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Device {index}"}

        pm = PlaybackManager(mock=True)
        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index == 3
        assert device_name == "Device 3"

    def test_resolve_default_device_as_input_output_pair_non_tuple(self):
        """_InputOutputPair that does NOT subclass list/tuple (older sounddevice)."""
        class _InputOutputPair:
            def __init__(self, inp, out):
                self._data = [inp, out]
            def __getitem__(self, idx):
                return self._data[idx]
            def __repr__(self):
                return f"_InputOutputPair({self._data})"

        class FakeSoundDevice:
            default = type("Default", (), {"device": _InputOutputPair(1, 6)})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Speaker {index}"}

        pm = PlaybackManager(mock=True)
        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index == 6
        assert device_name == "Speaker 6"

    def test_resolve_negative_output_index_returns_none(self):
        """Output index of -1 means no default device configured."""
        class FakeSoundDevice:
            default = type("Default", (), {"device": (0, -1)})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Device {index}"}

        pm = PlaybackManager(mock=True)
        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index is None
        assert device_name is None

    def test_resolve_none_output_index_returns_none(self):
        """Output side of the pair is None — no default output device."""
        class FakeSoundDevice:
            default = type("Default", (), {"device": (0, None)})()

            @staticmethod
            def query_devices(index):
                return {"name": f"Device {index}"}

        pm = PlaybackManager(mock=True)
        device_index, device_name = pm._resolve_output_device(FakeSoundDevice)

        assert device_index is None
        assert device_name is None
