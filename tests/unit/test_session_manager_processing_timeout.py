"""Regression tests for session timeout while stuck in PROCESSING."""

import time

from app.audio.session_manager import SessionManager
from app.utils.contracts import SessionState


def test_processing_state_times_out():
    """A session waiting for STT/processing must not hang forever."""
    sm = SessionManager(session_timeout_sec=1)
    sm.on_wake_detected()
    assert sm.state == SessionState.LISTENING

    sm.on_speech_end()
    assert sm.state == SessionState.PROCESSING

    time.sleep(1.5)
    assert sm.state == SessionState.IDLE


def test_listening_state_still_times_out():
    """The existing LISTENING timeout behavior must remain intact."""
    sm = SessionManager(session_timeout_sec=1)
    sm.on_wake_detected()
    assert sm.state == SessionState.LISTENING

    time.sleep(1.5)
    assert sm.state == SessionState.IDLE
