"""
Navigator - Wake Word Detector
Phase 3, Step 3.2

Wraps openWakeWord to provide always-on detection of the "hey navigator" phrase.
Operates in a background thread during idle state.

Behavior:
- Runs continuously while in idle mode
- Fires an activation callback when the configured phrase is detected
- Has a cooldown period to prevent double-triggers
- Stops cleanly when the session starts (active conversation)

Mock mode for testing:
    detector = WakeWordDetector(mock=True)
    detector.trigger()   # Manually fire the wake word callback
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Minimum seconds between two wake word activations
_COOLDOWN_SEC = 2.0


class WakeWordDetector:
    """
    Always-on wake word listener.
    Runs in a daemon thread and invokes on_activated() when triggered.

    Args:
        on_activated: Callback called (with no arguments) when wake word fires.
        mock:         In mock mode the detector does nothing autonomously —
                      call trigger() to simulate detection in tests.
    """

    def __init__(
        self,
        on_activated: Optional[Callable[[], None]] = None,
        mock: bool = False,
    ) -> None:
        cfg = get_settings()
        self._wake_word: str       = cfg.wake_word              # "hey navigator"
        self._threshold: float     = cfg.wake_word_threshold    # 0.5
        self._mock                 = mock
        self._on_activated         = on_activated or (lambda: None)
        self._running              = False
        self._active_session       = False  # set True while conversation is ongoing
        self._last_trigger_time    = 0.0
        self._thread: Optional[threading.Thread] = None
        self._oww_model            = None   # openWakeWord model (real mode)

        logger.info(
            "wakeword_init",
            phrase=self._wake_word,
            threshold=self._threshold,
            mock=self._mock,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, mic_frames_iter=None) -> None:
        """
        Start the wake word detection loop.

        Args:
            mic_frames_iter: Iterator of raw PCM frames from MicCapture.
                             Not required in mock mode.
        """
        if self._running:
            return
        self._running = True

        if self._mock:
            # No thread needed — trigger() is called manually in tests
            logger.info("wakeword_mock_started")
            return

        self._thread = threading.Thread(
            target=self._detect_loop,
            args=(mic_frames_iter,),
            daemon=True,
            name="wakeword-detect",
        )
        self._thread.start()
        logger.info("wakeword_started")

    def stop(self) -> None:
        """Stop the detection loop and release the model."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._oww_model = None
        logger.info("wakeword_stopped")

    def set_session_active(self, active: bool) -> None:
        """
        Suppress wake word triggers while a conversation session is open.
        Call with True when a session starts, False when it ends.
        """
        self._active_session = active

    def trigger(self) -> None:
        """
        Manually fire the activation callback.
        For use in tests and mock mode only.
        """
        self._fire_activation()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fire_activation(self) -> None:
        """Check cooldown and call the activation callback."""
        now = time.monotonic()
        if self._active_session:
            logger.debug("wakeword_suppressed_active_session")
            return
        if now - self._last_trigger_time < _COOLDOWN_SEC:
            logger.debug("wakeword_suppressed_cooldown")
            return
        self._last_trigger_time = now
        logger.info("wakeword_activated", phrase=self._wake_word)
        try:
            self._on_activated()
        except Exception as exc:
            logger.error("wakeword_callback_error", error=str(exc))

    def _load_model(self) -> bool:
        """Load the openWakeWord model. Returns True if successful."""
        try:
            from openwakeword.model import Model  # type: ignore
            # Load the "hey_navigator" or closest available model.
            # openWakeWord ships pretrained models; "hey_mycroft" pattern used as base.
            # The model name must match an available .tflite file.
            self._oww_model = Model(
                wakeword_models=["hey_jarvis"],   # closest built-in to "hey navigator"
                inference_framework="tflite",
            )
            logger.info("wakeword_model_loaded")
            return True
        except Exception as exc:
            logger.error("wakeword_model_load_failed", error=str(exc))
            return False

    def _detect_loop(self, mic_frames_iter) -> None:
        """Main detection loop: feed mic frames through the OWW model."""
        import numpy as np

        if not self._load_model():
            logger.error("wakeword_loop_aborted_no_model")
            self._running = False
            return

        logger.info("wakeword_loop_start")
        for frame in mic_frames_iter:
            if not self._running:
                break

            try:
                # Convert bytes → int16 array
                audio_data = np.frombuffer(frame, dtype=np.int16)
                prediction = self._oww_model.predict(audio_data)

                # prediction is a dict: {model_name: score}
                for model_name, score in prediction.items():
                    if score >= self._threshold:
                        logger.debug("wakeword_score", model=model_name, score=round(score, 3))
                        self._fire_activation()
                        break

            except Exception as exc:
                logger.error("wakeword_predict_error", error=str(exc))

        logger.info("wakeword_loop_end")
