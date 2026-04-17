"""
Navigator - Silero VAD (Voice Activity Detector)
Phase 3, Step 3.3

Wraps Silero VAD to gate audio during an active conversation session.
Only passes frames where human speech is detected. Trims silence so
that nothing is streamed to Deepgram during quiet periods.

Behavior:
- Accepts raw PCM frames (bytes) from MicCapture
- Returns speech frames and emits speech_start / speech_end events
- Requires torch and silero-vad to be installed for real mode
- Mock mode can simulate speech segments for testing

Usage:
    vad = SileroVAD(on_speech_start=..., on_speech_end=...)
    vad.process(frame)   # call for each mic frame during active session
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# How many consecutive non-speech frames before we declare end-of-utterance
_END_OF_UTTERANCE_FRAMES = 20  # ~640 ms at 16 kHz with 512-sample frames


class SileroVAD:
    """
    Voice activity detector using the Silero VAD model.

    Emits speech_start and speech_end events around detected utterances.
    Between events, passes speech frames to the on_speech_frame callback.

    Args:
        on_speech_start:  Called when speech begins. No arguments.
        on_speech_end:    Called when the utterance ends. No arguments.
        on_speech_frame:  Called with each speech frame (bytes) during utterance.
        threshold:        Probability above which a frame is considered speech (0–1).
        mock:             If True, load no model. Use set_mock_speech() in tests.
    """

    def __init__(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end:   Optional[Callable[[], None]] = None,
        on_speech_frame: Optional[Callable[[bytes], None]] = None,
        threshold: float = 0.5,
        mock: bool = False,
    ) -> None:
        cfg = get_settings()
        self._sample_rate = cfg.mic_sample_rate
        self._threshold   = threshold
        self._mock        = mock

        self._on_speech_start = on_speech_start or (lambda: None)
        self._on_speech_end   = on_speech_end   or (lambda: None)
        self._on_speech_frame = on_speech_frame or (lambda _: None)

        # State
        self._in_speech            = False
        self._silent_frame_count   = 0

        # Silero model objects (real mode only)
        self._model  = None
        self._utils  = None

        # Mock state
        self._mock_is_speech = False

        if not self._mock:
            self._load_model()

        logger.info("vad_init", threshold=self._threshold, mock=self._mock)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame: bytes) -> None:
        """
        Process one audio frame.
        Call this for every mic frame during an active session.

        Args:
            frame: Raw PCM bytes (16-bit signed, mono, 16 kHz).
        """
        is_speech = self._classify(frame)

        if is_speech:
            self._silent_frame_count = 0
            if not self._in_speech:
                self._in_speech = True
                logger.debug("vad_speech_start")
                try:
                    self._on_speech_start()
                except Exception as exc:
                    logger.error("vad_speech_start_callback_error", error=str(exc))

            try:
                self._on_speech_frame(frame)
            except Exception as exc:
                logger.error("vad_frame_callback_error", error=str(exc))

        else:
            if self._in_speech:
                self._silent_frame_count += 1
                # Still emit frames during short silence (keeps STT streaming smooth)
                try:
                    self._on_speech_frame(frame)
                except Exception:
                    pass

                if self._silent_frame_count >= _END_OF_UTTERANCE_FRAMES:
                    self._in_speech = False
                    self._silent_frame_count = 0
                    logger.debug("vad_speech_end")
                    try:
                        self._on_speech_end()
                    except Exception as exc:
                        logger.error("vad_speech_end_callback_error", error=str(exc))

    def reset(self) -> None:
        """Reset VAD state between sessions."""
        self._in_speech = False
        self._silent_frame_count = 0
        if self._model:
            try:
                self._model.reset_states()
            except Exception:
                pass
        logger.debug("vad_reset")

    @property
    def in_speech(self) -> bool:
        """True if VAD is currently inside a detected utterance."""
        return self._in_speech

    # ── Mock controls (for testing) ───────────────────────────────────────────

    def set_mock_speech(self, speaking: bool) -> None:
        """
        Control the mock VAD speech state.
        Call with True to simulate speech, False to simulate silence.
        """
        self._mock_is_speech = speaking

    # ── Internal ──────────────────────────────────────────────────────────────

    def _classify(self, frame: bytes) -> bool:
        """Return True if this frame contains speech."""
        if self._mock:
            return self._mock_is_speech

        if self._model is None:
            # Model failed to load — pass everything through (fail open)
            return True

        try:
            import torch
            audio_tensor = torch.frombuffer(frame, dtype=torch.int16).float() / 32768.0
            prob = self._model(audio_tensor.unsqueeze(0), self._sample_rate).item()
            return prob >= self._threshold
        except Exception as exc:
            logger.warning("vad_classify_error", error=str(exc))
            return True  # fail open: pass the frame through

    def _load_model(self) -> None:
        """Load the Silero VAD model from torch hub."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model = model
            self._utils = utils
            logger.info("vad_model_loaded")
        except Exception as exc:
            logger.error("vad_model_load_failed", error=str(exc))
            self._model = None
