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
import wave
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PlaybackState(str, Enum):
    IDLE = "idle"
    PLAYING = "playing"
    STOPPED = "stopped"
    DONE = "done"


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
        cfg = get_settings()

        self._on_complete = on_complete or (lambda: None)
        self._on_barge_in = on_barge_in or (lambda: None)
        self._mock = mock
        self._speaker_device_index = cfg.speaker_device_index

        self._state = PlaybackState.IDLE
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_stream: Optional[Any] = None
        self._resolved_output_device_name: Optional[str] = None

        if not self._mock:
            self._log_output_device()

        logger.info(
            "playback_init",
            mock=self._mock,
            speaker_device_index=self._speaker_device_index,
            output_device_name=self._resolved_output_device_name,
        )

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

        self.stop()

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

    def stop(self) -> None:
        """
        Stop playback immediately.
        Can be called as barge-in OR as a regular stop.
        """
        stream = None
        with self._lock:
            if self._state != PlaybackState.PLAYING:
                return
            self._state = PlaybackState.STOPPED
            stream = self._active_stream

        self._stop_event.set()
        if stream is not None:
            try:
                stream.abort()
            except Exception as exc:
                logger.warning("playback_abort_failed", error=str(exc))

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        logger.info("playback_stopped")

    def cancel(self) -> None:
        """Alias for stop() - explicit cancel intent."""
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

    def play_test_tone(self, duration_ms: int = 300, frequency_hz: float = 440.0) -> None:
        """Play a tiny audible tone through the configured output device."""
        self.play(self._build_test_tone_wav(duration_ms=duration_ms, frequency_hz=frequency_hz))

    def _playback_loop(self, audio_bytes: bytes) -> None:
        """Run audio playback in the background thread."""
        if self._mock:
            self._mock_playback()
        else:
            self._real_playback(audio_bytes)

        with self._lock:
            if self._state == PlaybackState.PLAYING:
                self._state = PlaybackState.DONE
                logger.info("playback_done_naturally")
                try:
                    self._on_complete()
                except Exception as exc:
                    logger.error("playback_complete_callback_error", error=str(exc))

    def _mock_playback(self) -> None:
        """Simulate playback in mock mode - completes after 100 ms."""
        logger.info("playback_started", mock=True)
        time.sleep(0.1)
        if self._stop_event.is_set():
            with self._lock:
                self._state = PlaybackState.STOPPED

    def _real_playback(self, audio_bytes: bytes) -> None:
        """Play audio using sounddevice OutputStream for more reliable Windows playback."""
        try:
            import sounddevice as sd  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError as exc:
            logger.error("playback_backend_missing", error=str(exc))
            with self._lock:
                self._state = PlaybackState.STOPPED
            return

        stream = None
        try:
            buf = io.BytesIO(audio_bytes)
            data, samplerate = sf.read(buf, dtype="float32", always_2d=True)
            if data.size == 0:
                logger.error("playback_no_samples_decoded")
                with self._lock:
                    self._state = PlaybackState.STOPPED
                return

            device_index, device_name = self._resolve_output_device(sd)
            channels = int(data.shape[1])
            chunk_size = min(max(1024, samplerate // 10), max(len(data), 1024))

            stream = sd.OutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype="float32",
                device=device_index,
                blocksize=chunk_size,
            )
            stream.start()
            with self._lock:
                self._active_stream = stream

            logger.info(
                "playback_started",
                bytes=len(audio_bytes),
                samplerate=samplerate,
                channels=channels,
                device_index=device_index,
                output_device_name=device_name,
                mock=False,
            )

            for start in range(0, len(data), chunk_size):
                if self._stop_event.is_set():
                    logger.debug("playback_interrupted_during_stream")
                    with self._lock:
                        self._state = PlaybackState.STOPPED
                    return

                end = min(start + chunk_size, len(data))
                stream.write(data[start:end])

        except Exception as exc:
            logger.error(
                "playback_device_open_failed",
                error=str(exc),
                speaker_device_index=self._speaker_device_index,
                output_device_name=self._resolved_output_device_name,
            )
            with self._lock:
                self._state = PlaybackState.STOPPED
        finally:
            with self._lock:
                self._active_stream = None
            if stream is not None:
                try:
                    stream.stop()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass

    def _log_output_device(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.warning("playback_device_probe_skipped", reason="sounddevice_not_installed")
            return

        try:
            device_index, device_name = self._resolve_output_device(sd)
            logger.info(
                "playback_output_device_resolved",
                output_device_index=device_index,
                output_device_name=device_name,
                explicit_device_selected=self._speaker_device_index is not None,
            )
        except Exception as exc:
            logger.error(
                "playback_output_device_probe_failed",
                error=str(exc),
                speaker_device_index=self._speaker_device_index,
            )

    def _resolve_output_device(self, sounddevice_module: Any) -> tuple[Optional[int], Optional[str]]:
        if self._speaker_device_index is not None:
            device_index = int(self._speaker_device_index)
            device_name = self._query_device_name(sounddevice_module, device_index)
            self._resolved_output_device_name = device_name
            return device_index, device_name

        raw = sounddevice_module.default.device
        # sounddevice.default.device can be:
        #   - int: the output device index directly
        #   - list/tuple: [input_idx, output_idx]
        #   - _InputOutputPair: custom class that may NOT subclass list/tuple depending
        #     on the sounddevice version installed — output index is always at position 1
        if isinstance(raw, (list, tuple)):
            raw_output = raw[1]
        elif hasattr(raw, "__getitem__"):
            try:
                raw_output = raw[1]
            except (IndexError, TypeError):
                raw_output = raw
        else:
            raw_output = raw

        if raw_output is None:
            self._resolved_output_device_name = None
            return None, None

        try:
            output_index = int(raw_output)
        except (TypeError, ValueError) as exc:
            logger.error(
                "playback_output_device_resolution_failed",
                raw_device=repr(raw),
                raw_output=repr(raw_output),
                error=str(exc),
            )
            self._resolved_output_device_name = None
            return None, None

        if output_index < 0:
            self._resolved_output_device_name = None
            return None, None

        device_name = self._query_device_name(sounddevice_module, output_index)
        self._resolved_output_device_name = device_name
        return output_index, device_name

    @staticmethod
    def _query_device_name(sounddevice_module: Any, device_index: int) -> Optional[str]:
        info = sounddevice_module.query_devices(device_index)
        if isinstance(info, dict):
            return str(info.get("name") or "").strip() or None
        return None

    @staticmethod
    def _build_test_tone_wav(duration_ms: int = 300, frequency_hz: float = 440.0) -> bytes:
        import numpy as np  # type: ignore

        sample_rate = 22050
        duration_sec = max(duration_ms, 50) / 1000.0
        timeline = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        waveform = 0.2 * np.sin(2.0 * np.pi * frequency_hz * timeline)
        pcm16 = (waveform * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buf.getvalue()
