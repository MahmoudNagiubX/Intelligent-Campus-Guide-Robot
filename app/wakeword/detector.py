"""
Navigator - Wake Word Detector
Phase 3, Step 3.2

Wraps openWakeWord to provide always-on detection of the configured wake
phrase.

Behavior:
- Runs continuously while in idle mode
- Fires an activation callback when the configured phrase is detected
- Has a cooldown period to prevent double-triggers
- Can also process frames inline when the runtime owns the mic loop

Mock mode for testing:
    detector = WakeWordDetector(mock=True)
    detector.trigger()
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
import re
import sys
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


_COOLDOWN_SEC = 2.0


class WakeWordDetector:
    """
    Always-on wake word listener.

    Args:
        on_activated: Callback called when wake word fires.
        mock: If True, no model is loaded and trigger() is used in tests.
    """

    def __init__(
        self,
        on_activated: Optional[Callable[[], None]] = None,
        mock: bool = False,
    ) -> None:
        cfg = get_settings()
        self._wake_word: str = cfg.wake_word
        self._wake_word_model_ref: str = self._resolve_model_reference(cfg.wake_word_model, self._wake_word)
        self._inference_framework: str = self._resolve_inference_framework(self._wake_word_model_ref)
        self._threshold: float = cfg.wake_word_threshold
        self._mock = mock
        self._on_activated = on_activated or (lambda: None)
        self._running = False
        self._active_session = False
        self._last_trigger_time = 0.0
        self._thread: Optional[threading.Thread] = None
        self._oww_model = None

        logger.info(
            "wakeword_init",
            phrase=self._wake_word,
            model=self._wake_word_model_ref,
            framework=self._inference_framework,
            threshold=self._threshold,
            mock=self._mock,
        )

    def start(self, mic_frames_iter=None) -> None:
        """
        Start the background wake-word loop.

        The live Pipecat runtime usually feeds process_frame() directly instead.
        """
        if self._running:
            return
        self._running = True

        if self._mock:
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
        """Stop detection and release the loaded model."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._oww_model = None
        logger.info("wakeword_stopped")

    def set_session_active(self, active: bool) -> None:
        """Suppress activations while a conversation is already active."""
        self._active_session = active

    def trigger(self) -> None:
        """Manually fire the activation callback."""
        self._fire_activation()

    def process_frame(self, frame: bytes) -> None:
        """
        Process one PCM frame inline.

        This is used by the Phase 8 runtime so the mic stream can be shared
        between wake-word detection and the rest of the live pipeline.
        """
        if self._mock:
            return

        if self._oww_model is None and not self._load_model():
            return

        try:
            prediction = self._predict_frame(frame)
            for model_name, score in prediction.items():
                if score >= self._threshold:
                    logger.debug("wakeword_score", model=model_name, score=round(score, 3))
                    self._fire_activation()
                    break
        except Exception as exc:
            logger.error("wakeword_predict_error", error=str(exc))

    def _fire_activation(self) -> None:
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
        """Load the openWakeWord model."""
        try:
            import openwakeword
            from openwakeword.model import Model  # type: ignore

            if not self._is_model_path(self._wake_word_model_ref):
                self._validate_builtin_model_name(
                    self._wake_word_model_ref,
                    set(openwakeword.MODELS.keys()),
                    self._wake_word,
                )
                try:
                    openwakeword.utils.download_models([self._wake_word_model_ref])
                except Exception as exc:
                    logger.debug(
                        "wakeword_model_download_skipped",
                        model=self._wake_word_model_ref,
                        error=str(exc),
                    )

            self._oww_model = Model(
                wakeword_models=[self._wake_word_model_ref],
                inference_framework=self._inference_framework,
            )
            logger.info(
                "wakeword_model_loaded",
                phrase=self._wake_word,
                model=self._wake_word_model_ref,
                framework=self._inference_framework,
            )
            return True
        except Exception as exc:
            logger.error(
                "wakeword_model_load_failed",
                phrase=self._wake_word,
                model=self._wake_word_model_ref,
                framework=self._inference_framework,
                error=str(exc),
            )
            return False

    def _predict_frame(self, frame: bytes) -> dict:
        """Run one frame through the wake-word model."""
        import numpy as np

        if self._oww_model is None:
            return {}

        audio_data = np.frombuffer(frame, dtype=np.int16)
        return self._oww_model.predict(audio_data)

    def _detect_loop(self, mic_frames_iter) -> None:
        """Main background detection loop."""
        if not self._load_model():
            logger.error("wakeword_loop_aborted_no_model")
            self._running = False
            return

        logger.info("wakeword_loop_start")
        for frame in mic_frames_iter:
            if not self._running:
                break
            self.process_frame(frame)

        logger.info("wakeword_loop_end")

    @staticmethod
    def _resolve_model_reference(configured_model: str, wake_word: str) -> str:
        model_ref = configured_model.strip()
        if model_ref:
            return model_ref
        normalized = re.sub(r"[^a-z0-9]+", "_", wake_word.strip().lower())
        return normalized.strip("_")

    @staticmethod
    def _is_model_path(model_ref: str) -> bool:
        suffix = Path(model_ref).suffix.lower()
        return suffix in {".onnx", ".tflite"}

    @classmethod
    def _resolve_inference_framework(cls, model_ref: str) -> str:
        suffix = Path(model_ref).suffix.lower()
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".tflite":
            return "tflite"
        return "tflite" if sys.platform.startswith("linux") else "onnx"

    @classmethod
    def _validate_builtin_model_name(
        cls,
        model_ref: str,
        available_models: set[str],
        wake_word: str,
    ) -> None:
        if cls._is_model_path(model_ref):
            return
        if model_ref in available_models:
            return

        supported = ", ".join(sorted(available_models))
        raise ValueError(
            f"No built-in openWakeWord model named '{model_ref}' for wake phrase '{wake_word}'. "
            f"Supported built-in models: {supported}. "
            "For custom phrases, set WAKE_WORD_MODEL to a local .onnx or .tflite file."
        )
