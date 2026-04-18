"""
Navigator - Phase 3 Step 3.5: Local Listening Stack Tests
Tests for MicCapture (mock mode), WakeWordDetector, SileroVAD, and SessionManager.

All tests run without real audio hardware (mock mode) and without
openWakeWord or Silero models being installed.

Run with:
    pytest tests/unit/test_listening_stack.py -v
"""

import threading
import time

import pytest

from app.audio.mic_input import MicCapture
from app.config.settings import get_settings
from app.audio.session_manager import SessionManager
from app.utils.contracts import SessionState
from app.vad.silero_vad import SileroVAD
from app.wakeword.detector import WakeWordDetector


# ─────────────────────────────────────────────────────────────────────────────
# MicCapture (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestMicCaptureMock:
    def test_mock_mode_starts_and_produces_frames(self):
        mic = MicCapture(mock=True)
        mic.start()
        frames = []
        for frame in mic.frames():
            frames.append(frame)
            if len(frames) >= 5:
                mic.stop()
                break
        assert len(frames) == 5

    def test_mock_frames_are_bytes(self):
        mic = MicCapture(mock=True)
        mic.start()
        frame = next(mic.frames())
        mic.stop()
        assert isinstance(frame, bytes)

    def test_mock_frame_length_matches_config(self, monkeypatch):
        from app.config import get_settings
        cfg = get_settings()
        mic = MicCapture(mock=True)
        mic.start()
        frame = next(mic.frames())
        mic.stop()
        # 16-bit (2 bytes per sample) × frame_size samples
        expected_bytes = cfg.mic_frame_size * 2
        assert len(frame) == expected_bytes

    def test_mock_stop_drains_queue(self):
        mic = MicCapture(mock=True)
        mic.start()
        time.sleep(0.05)   # let a few frames accumulate
        mic.stop()
        # After stop, frames() should eventually exhaust
        drained = list(mic.frames())
        # We just verify it terminates without hanging
        assert isinstance(drained, list)

    def test_sample_rate_property(self):
        mic = MicCapture(mock=True)
        assert mic.sample_rate == 16000

    def test_frame_size_property(self):
        from app.config import get_settings
        mic = MicCapture(mock=True)
        assert mic.frame_size == get_settings().mic_frame_size


# ─────────────────────────────────────────────────────────────────────────────
# WakeWordDetector (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestWakeWordDetectorMock:
    def test_mock_callback_fires_on_trigger(self):
        fired = []
        detector = WakeWordDetector(on_activated=lambda: fired.append(True), mock=True)
        detector.start()
        detector.trigger()
        assert len(fired) == 1

    def test_multiple_triggers_respect_cooldown(self):
        fired = []
        detector = WakeWordDetector(on_activated=lambda: fired.append(True), mock=True)
        detector.start()
        detector.trigger()
        detector.trigger()  # within cooldown, should be suppressed
        assert len(fired) == 1

    def test_no_activation_while_session_active(self):
        fired = []
        detector = WakeWordDetector(on_activated=lambda: fired.append(True), mock=True)
        detector.start()
        detector.set_session_active(True)
        detector.trigger()
        assert len(fired) == 0

    def test_activation_after_session_ends(self):
        fired = []
        detector = WakeWordDetector(on_activated=lambda: fired.append(True), mock=True)
        detector.start()
        detector.set_session_active(True)
        detector.trigger()     # suppressed
        detector.set_session_active(False)
        # Force cooldown to expire by manipulating last trigger time
        detector._last_trigger_time = 0.0
        detector.trigger()     # should fire now
        assert len(fired) == 1

    def test_no_callback_set_does_not_crash(self):
        detector = WakeWordDetector(mock=True)
        detector.start()
        detector.trigger()   # no callback set — must not raise

    def test_stop_does_not_crash(self):
        detector = WakeWordDetector(mock=True)
        detector.start()
        detector.stop()


class TestWakeWordDetectorConfig:
    def test_model_reference_defaults_from_phrase(self, monkeypatch):
        monkeypatch.setenv("WAKE_WORD", "hey jarvis")
        monkeypatch.delenv("WAKE_WORD_MODEL", raising=False)
        get_settings.cache_clear()
        try:
            detector = WakeWordDetector(mock=True)
            assert detector._wake_word == "hey jarvis"
            assert detector._wake_word_model_ref == "hey_jarvis"
        finally:
            get_settings.cache_clear()

    def test_explicit_model_path_overrides_phrase(self, monkeypatch):
        monkeypatch.setenv("WAKE_WORD", "hey robot")
        monkeypatch.setenv("WAKE_WORD_MODEL", "./data/wakewords/robot.onnx")
        get_settings.cache_clear()
        try:
            detector = WakeWordDetector(mock=True)
            assert detector._wake_word_model_ref == "./data/wakewords/robot.onnx"
            assert detector._inference_framework == "onnx"
        finally:
            get_settings.cache_clear()

    def test_invalid_builtin_model_name_has_actionable_error(self):
        with pytest.raises(ValueError) as excinfo:
            WakeWordDetector._validate_builtin_model_name(
                "hey_robot",
                {"hey_jarvis", "hey_mycroft", "hey_rhasspy"},
                "hey robot",
            )

        message = str(excinfo.value)
        assert "hey_robot" in message
        assert "hey robot" in message
        assert "WAKE_WORD_MODEL" in message
        assert "hey_jarvis" in message


# ─────────────────────────────────────────────────────────────────────────────
# SileroVAD (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestSileroVADMock:
    def _make_vad(self, **kwargs):
        return SileroVAD(mock=True, **kwargs)

    def test_no_event_for_silence(self):
        started = []
        vad = self._make_vad(on_speech_start=lambda: started.append(True))
        vad.set_mock_speech(False)
        vad.process(b"\x00" * 1024)
        assert started == []

    def test_speech_start_event_fires(self):
        started = []
        vad = self._make_vad(on_speech_start=lambda: started.append(True))
        vad.set_mock_speech(True)
        vad.process(b"\x00" * 1024)
        assert len(started) == 1

    def test_speech_start_fires_only_once_per_utterance(self):
        started = []
        vad = self._make_vad(on_speech_start=lambda: started.append(True))
        vad.set_mock_speech(True)
        for _ in range(5):
            vad.process(b"\x00" * 1024)
        assert len(started) == 1

    def test_speech_frame_callback_called_during_speech(self):
        frames = []
        vad = self._make_vad(on_speech_frame=lambda f: frames.append(f))
        vad.set_mock_speech(True)
        for _ in range(3):
            vad.process(b"\x01" * 1024)
        assert len(frames) == 3

    def test_speech_end_after_silence_frames(self):
        from app.vad.silero_vad import _END_OF_UTTERANCE_FRAMES
        ended = []
        vad = self._make_vad(on_speech_end=lambda: ended.append(True))

        # Start speech
        vad.set_mock_speech(True)
        vad.process(b"\x00" * 1024)
        assert vad.in_speech is True

        # End speech — need N silence frames
        vad.set_mock_speech(False)
        for _ in range(_END_OF_UTTERANCE_FRAMES):
            vad.process(b"\x00" * 1024)

        assert len(ended) == 1
        assert vad.in_speech is False

    def test_reset_clears_speech_state(self):
        vad = self._make_vad()
        vad.set_mock_speech(True)
        vad.process(b"\x00" * 1024)
        assert vad.in_speech is True
        vad.reset()
        assert vad.in_speech is False


# ─────────────────────────────────────────────────────────────────────────────
# SessionManager
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionManager:
    def _make_sm(self, timeout: int = 30) -> SessionManager:
        return SessionManager(session_timeout_sec=timeout)

    def test_initial_state_is_idle(self):
        sm = self._make_sm()
        assert sm.state == SessionState.IDLE

    def test_wake_detected_transition(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        # Should be LISTENING (wake_detected → listening happens immediately)
        assert sm.state == SessionState.LISTENING

    def test_session_id_created_on_wake(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        assert sm.session_id is not None

    def test_speech_end_moves_to_processing(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_speech_end()
        assert sm.state == SessionState.PROCESSING

    def test_response_ready_moves_to_speaking(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_speech_end()
        sm.on_response_ready()
        assert sm.state == SessionState.SPEAKING

    def test_playback_complete_returns_to_idle(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_speech_end()
        sm.on_response_ready()
        sm.on_playback_complete()
        assert sm.state == SessionState.IDLE
        assert sm.session_id is None

    def test_barge_in_moves_to_listening(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_speech_end()
        sm.on_response_ready()
        # Now in SPEAKING — barge in
        sm.on_barge_in()
        assert sm.state == SessionState.LISTENING

    def test_error_from_any_state(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_error("test error")
        assert sm.state == SessionState.ERROR

    def test_reset_returns_to_idle(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_error("xyz")
        sm.reset()
        assert sm.state == SessionState.IDLE
        assert sm.session_id is None

    def test_invalid_transition_returns_false(self):
        sm = self._make_sm()
        # Can't go IDLE → SPEAKING
        result = sm.transition(SessionState.SPEAKING)
        assert result is False
        assert sm.state == SessionState.IDLE   # unchanged

    def test_double_wake_in_active_session_ignored(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        assert sm.state == SessionState.LISTENING
        sm.on_wake_detected()   # should log a warning and be ignored
        assert sm.state == SessionState.LISTENING

    def test_session_timeout_resets_to_idle(self):
        timeout_fired = []
        sm = SessionManager(session_timeout_sec=1, on_timeout=lambda: timeout_fired.append(True))
        sm.on_wake_detected()
        sm.start_timeout_timer()
        time.sleep(1.3)
        assert sm.state == SessionState.IDLE
        assert timeout_fired == [True]

    def test_all_transitions_logged_without_crash(self):
        sm = self._make_sm()
        sm.on_wake_detected()
        sm.on_speech_end()
        sm.on_response_ready()
        sm.on_playback_complete()
        sm.on_wake_detected()
        sm.on_barge_in()
        assert sm.state == SessionState.LISTENING
