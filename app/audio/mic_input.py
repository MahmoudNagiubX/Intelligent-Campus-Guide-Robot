"""
Navigator - Microphone Capture Service
Phase 3, Step 3.1

Captures raw PCM audio from the system microphone using PyAudio.
Outputs frames at 16 kHz, mono (1 channel), 16-bit signed int.
This is the only place PyAudio is used — all downstream modules
receive raw PCM bytes, not PyAudio objects.

Usage:
    mic = MicCapture()
    mic.start()
    for frame in mic.frames():
        process(frame)
    mic.stop()

Mock mode (for testing without hardware):
    mic = MicCapture(mock=True)
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from queue import Empty, Queue
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class MicCapture:
    """
    Captures audio from the microphone in a background thread.
    Produces fixed-size PCM frames (bytes) via the frames() generator.

    Args:
        device_index: Override the system-default mic device.
                      Pass None to use the system default.
        mock:         If True, produce synthetic silence frames
                      (for unit tests and CI without audio hardware).
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        mock: bool = False,
    ) -> None:
        cfg = get_settings()
        self._sample_rate: int = cfg.mic_sample_rate        # Hz (16000)
        self._frame_size: int  = cfg.mic_frame_size         # samples per frame (512)
        self._channels: int    = cfg.mic_channels           # 1 = mono
        self._device_index     = device_index or cfg.mic_device_index
        self._mock             = mock

        self._queue: Queue[bytes] = Queue(maxsize=512)
        self._running            = False
        self._thread: Optional[threading.Thread] = None
        self._stream             = None   # PyAudio stream (real mode only)
        self._pa                 = None   # PyAudio instance

        logger.info(
            "mic_init",
            sample_rate=self._sample_rate,
            frame_size=self._frame_size,
            device_index=self._device_index,
            mock=self._mock,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Open the mic device and begin capturing audio in a background thread."""
        if self._running:
            return
        self._running = True

        if self._mock:
            self._thread = threading.Thread(
                target=self._mock_loop, daemon=True, name="mic-mock"
            )
        else:
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="mic-capture"
            )

        self._thread.start()
        logger.info("mic_started", mock=self._mock)

    def stop(self) -> None:
        """Stop capturing and release hardware resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_stream()
        logger.info("mic_stopped")

    def frames(self) -> Generator[bytes, None, None]:
        """
        Generator that yields PCM audio frames.
        Block-waits up to 100 ms per frame.
        Stops when stop() is called and the queue is drained.
        """
        while self._running or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
                yield frame
            except Empty:
                continue

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        return self._frame_size

    # ── Internal capture loop (real hardware) ────────────────────────────────

    def _capture_loop(self) -> None:
        """Open PyAudio stream and push frames into the queue."""
        try:
            import pyaudio  # imported here so mock mode never needs it
        except ImportError:
            logger.error("mic_pyaudio_not_installed")
            self._running = False
            return

        self._pa = pyaudio.PyAudio()

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                frames_per_buffer=self._frame_size,
                input_device_index=self._device_index,
            )
            logger.info("mic_stream_open", device_index=self._device_index)
        except OSError as exc:
            logger.error("mic_open_failed", error=str(exc))
            self._running = False
            self._close_stream()
            return

        while self._running:
            try:
                data = self._stream.read(self._frame_size, exception_on_overflow=False)
                if not self._queue.full():
                    self._queue.put_nowait(data)
            except Exception as exc:
                logger.error("mic_read_error", error=str(exc))
                break

        self._close_stream()

    def _close_stream(self) -> None:
        """Safely close PyAudio stream and terminate the PA instance."""
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
        except Exception:
            pass
        try:
            if self._pa:
                self._pa.terminate()
                self._pa = None
        except Exception:
            pass

    # ── Mock loop (testing / CI without hardware) ─────────────────────────────

    def _mock_loop(self) -> None:
        """Generate synthetic silence frames at the correct timing."""
        # Duration of one frame in seconds
        frame_duration = self._frame_size / self._sample_rate
        # 16-bit silence = 2 bytes per sample
        silence = bytes(self._frame_size * 2)

        while self._running:
            if not self._queue.full():
                self._queue.put_nowait(silence)
            time.sleep(frame_duration)
