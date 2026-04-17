"""
Navigator - Audio Playback Manager with Barge-In
Phase 7, Steps 7.2 + 7.3

Manages audio playback from TTS synthesis.
Implements barge-in: if the user speaks during playback, it stops immediately.

Behavior:
- play(audio_bytes): starts playback in a background thread
- stop(): stops playback immediately (barge-in)
- cancel(): same as stop, but marks as cancelled (not completed)
- on_complete callback: fired when playback finishes naturally
- on_barge_in callback: call this from the VAD when speech is detected

State:
    IDLE     -> not playing
    PLAYING  -> audio is being played
    STOPPED  -> stopped by stop() or cancel()
    DONE     -> finished naturally

Mock mode:
    player = PlaybackManager(mock=True)
    player is schedulable but completes immediately without real audio
"""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


class PlaybackState(str, Enum):
    IDLE    = "idle"
    PLAYING = "playing"
    STOPPED = "stopped"
    DONE    = "done"


class PlaybackManager:
    """
    Manages audio playback with barge-in support.

    Args:
        on_complete: Called when playback finishes naturally (not interrupted).
        on_barge_in: Called if external code calls notify_speech_detected()
                     during playback (i.e., user interrupted the robot).
        mock:        If True, "play" completes in mock time without real audio.
    """

    def __init__(
        self,
        on_complete: Optional[Callable[[], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
        mock: bool = False,
    ) -> None:
        self._on_complete = on_complete or (lambda: None)
        self._on_barge_in = on_barge_in or (lambda: None)
        self._mock        = mock

        self._state         = PlaybackState.IDLE
        self._lock          = threading.Lock()
        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        logger.info("playback_init", mock=self._mock)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> PlaybackState:
        with self._lock:
            return self._state

    @property
    def is_playing(self) -> bool:
        return self.state == PlaybackState.PLAYING

    def play(self, audio_bytes: bytes) -> None:
        """
        Start audio playback in a background thread.
        If audio is already playing, stop it first.

        Args:
            audio_bytes: Raw audio bytes (MP3 or WAV).
        """
        if not audio_bytes:
            logger.warning("playback_empty_audio_skipped")
            return

        self.stop()   # cancel any ongoing playback

        self._stop_event.clear()
        with self._lock:
            self._state = PlaybackState.PLAYING

        self._thread = threading.Thread(
            target=self._playback_loop,
            args=(audio_bytes,),
            daemon=True,
            name="tts-playback",
        )
        self._thread.start()
        logger.info("playback_started", bytes=len(audio_bytes))

    def stop(self) -> None:
        """
        Stop playback immediately.
        Can be called as barge-in OR as a regular stop.
        """
        with self._lock:
            if self._state != PlaybackState.PLAYING:
                return
            self._state = PlaybackState.STOPPED

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("playback_stopped")

    def cancel(self) -> None:
        """Alias for stop() — explicit cancel intent."""
        self.stop()

    def notify_speech_detected(self) -> None:
        """
        Call this from the VAD when speech is detected during playback.
        This is the barge-in trigger.
        """
        if self.state == PlaybackState.PLAYING:
            logger.info("playback_barge_in_detected")
            self.stop()
            try:
                self._on_barge_in()
            except Exception as exc:
                logger.error("playback_barge_in_callback_error", error=str(exc))

    # ── Playback loop ─────────────────────────────────────────────────────────

    def _playback_loop(self, audio_bytes: bytes) -> None:
        """Run audio playback in the background thread."""
        if self._mock:
            self._mock_playback()
        else:
            self._real_playback(audio_bytes)

        with self._lock:
            if self._state == PlaybackState.PLAYING:
                # Completed naturally
                self._state = PlaybackState.DONE
                logger.info("playback_done_naturally")
                try:
                    self._on_complete()
                except Exception as exc:
                    logger.error("playback_complete_callback_error", error=str(exc))

    def _mock_playback(self) -> None:
        """Simulate playback in mock mode — completes after 100 ms."""
        time.sleep(0.1)
        if self._stop_event.is_set():
            with self._lock:
                self._state = PlaybackState.STOPPED

    def _real_playback(self, audio_bytes: bytes) -> None:
        """Play audio using sounddevice and soundfile, streaming chunk by chunk."""
        try:
            import sounddevice as sd   # type: ignore
            import soundfile as sf     # type: ignore
        except ImportError:
            logger.error("playback_sounddevice_not_installed")
            with self._lock:
                self._state = PlaybackState.STOPPED
            return

        try:
            buf = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(buf, dtype="float32")

            chunk_size = 1024
            start = 0

            while start < len(data):
                if self._stop_event.is_set():
                    logger.debug("playback_interrupted_during_stream")
                    with self._lock:
                        self._state = PlaybackState.STOPPED
                    return

                end = min(start + chunk_size, len(data))
                chunk = data[start:end]
                sd.play(chunk, samplerate=samplerate, blocking=True)
                start = end

        except Exception as exc:
            logger.error("playback_error", error=str(exc))
            with self._lock:
                self._state = PlaybackState.STOPPED
