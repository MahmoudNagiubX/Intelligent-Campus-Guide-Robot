"""
Unit tests for SessionManager multi-turn keepalive behavior.

Key invariants tested:
- wake → listening
- speech_start resets activity
- speech_end enters processing
- response_ready enters speaking
- playback_complete returns to LISTENING (not IDLE)
- inactivity timeout returns to IDLE
- empty_response returns to LISTENING (not IDLE)
"""

import time

import pytest

from app.audio.session_manager import SessionManager
from app.utils.contracts import SessionState


def _sm(timeout: int = 5) -> SessionManager:
    return SessionManager(session_timeout_sec=timeout)


# ------------------------------------------------------------------
# Basic lifecycle
# ------------------------------------------------------------------


def test_wake_transitions_to_listening():
    sm = _sm()
    sm.on_wake_detected()
    assert sm.state == SessionState.LISTENING
    assert sm.session_id is not None


def test_speech_start_resets_activity_in_listening():
    sm = _sm()
    sm.on_wake_detected()
    before = sm._last_activity
    time.sleep(0.02)
    sm.on_speech_start()
    assert sm._last_activity > before
    assert sm.state == SessionState.LISTENING


def test_speech_end_enters_processing():
    sm = _sm()
    sm.on_wake_detected()
    sm.on_speech_end()
    assert sm.state == SessionState.PROCESSING


def test_response_ready_enters_speaking():
    sm = _sm()
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_response_ready()
    assert sm.state == SessionState.SPEAKING


# ------------------------------------------------------------------
# Keepalive: playback_complete -> LISTENING (not IDLE)
# ------------------------------------------------------------------


def test_playback_complete_returns_to_listening_not_idle():
    sm = _sm()
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_response_ready()
    sm.on_playback_complete()
    assert sm.state == SessionState.LISTENING, (
        "After TTS playback, session must stay active (LISTENING), not go to IDLE"
    )


def test_playback_complete_keeps_session_id():
    sm = _sm()
    sm.on_wake_detected()
    original_id = sm.session_id
    sm.on_speech_end()
    sm.on_response_ready()
    sm.on_playback_complete()
    assert sm.session_id == original_id


def test_playback_complete_restarts_timeout_timer():
    sm = _sm(timeout=1)
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_response_ready()
    sm.on_playback_complete()

    assert sm.state == SessionState.LISTENING
    # Timer is armed — after timeout, session should go to IDLE
    time.sleep(1.5)
    assert sm.state == SessionState.IDLE


# ------------------------------------------------------------------
# Multi-turn: two questions without re-wake
# ------------------------------------------------------------------


def test_multi_turn_same_session():
    sm = _sm()
    sm.on_wake_detected()
    session_id = sm.session_id

    # First turn
    sm.on_speech_end()
    sm.on_response_ready()
    sm.on_playback_complete()
    assert sm.state == SessionState.LISTENING
    assert sm.session_id == session_id

    # Second turn — no re-wake needed
    sm.on_speech_start()
    sm.on_speech_end()
    sm.on_response_ready()
    sm.on_playback_complete()
    assert sm.state == SessionState.LISTENING
    assert sm.session_id == session_id


# ------------------------------------------------------------------
# Empty response (noise/unclear): stays in LISTENING
# ------------------------------------------------------------------


def test_empty_response_returns_to_listening():
    sm = _sm()
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_empty_response()
    assert sm.state == SessionState.LISTENING


def test_empty_response_keeps_session_id():
    sm = _sm()
    sm.on_wake_detected()
    sid = sm.session_id
    sm.on_speech_end()
    sm.on_empty_response()
    assert sm.session_id == sid


# ------------------------------------------------------------------
# Timeout: inactivity returns to IDLE
# ------------------------------------------------------------------


def test_timeout_returns_to_idle():
    sm = _sm(timeout=1)
    sm.on_wake_detected()
    assert sm.state == SessionState.LISTENING
    time.sleep(1.5)
    assert sm.state == SessionState.IDLE


def test_activity_ping_delays_timeout():
    timeout_fired = []

    def _on_timeout():
        timeout_fired.append(True)

    sm = SessionManager(session_timeout_sec=1, on_timeout=_on_timeout)
    sm.on_wake_detected()

    # Ping at 0.7s — should delay the 1s timer
    time.sleep(0.7)
    sm.activity_ping()
    time.sleep(0.7)
    # Only 0.7s since the ping, so timeout should NOT have fired yet
    assert not timeout_fired, "activity_ping should have reset the inactivity timer"


# ------------------------------------------------------------------
# Session end via end_session returns to IDLE
# ------------------------------------------------------------------


def test_end_session_from_listening_returns_idle():
    sm = _sm()
    sm.on_wake_detected()
    sm.end_session(reason="manual")
    assert sm.state == SessionState.IDLE
    assert sm.session_id is None


# ------------------------------------------------------------------
# Barge-in during speaking
# ------------------------------------------------------------------


def test_barge_in_during_speaking():
    sm = _sm()
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_response_ready()
    assert sm.state == SessionState.SPEAKING

    sm.on_barge_in()
    assert sm.state == SessionState.LISTENING
