"""Tests for playback echo suppression after TTS output."""

import time

from app.config.settings import get_settings
from app.tts.playback import PlaybackManager, PlaybackState


def test_barge_in_suppressed_immediately_after_playback():
    """Barge-in must be ignored inside the echo suppression window."""
    barged = []
    pm = PlaybackManager(mock=True, on_barge_in=lambda: barged.append(True))
    pm._echo_suppress_until = time.monotonic() + 1.5

    pm.notify_speech_detected()

    assert barged == []


def test_barge_in_works_after_suppress_window():
    """After the suppress window, real barge-in must still work."""
    barged = []
    pm = PlaybackManager(mock=True, on_barge_in=lambda: barged.append(True))
    pm._echo_suppress_until = time.monotonic() - 1.0
    pm._state = PlaybackState.PLAYING

    pm.notify_speech_detected()

    assert len(barged) == 1


def test_is_echo_suppressed_returns_correctly():
    pm = PlaybackManager(mock=True)
    pm._echo_suppress_until = time.monotonic() + 5.0
    assert pm.is_echo_suppressed() is True

    pm._echo_suppress_until = time.monotonic() - 1.0
    assert pm.is_echo_suppressed() is False


def test_echo_suppress_ms_configurable(monkeypatch):
    """Echo suppress duration should come from settings."""
    monkeypatch.setenv("PLAYBACK_ECHO_SUPPRESS_MS", "500.0")
    get_settings.cache_clear()
    try:
        cfg = get_settings()
        assert cfg.playback_echo_suppress_ms == 500.0
    finally:
        get_settings.cache_clear()
