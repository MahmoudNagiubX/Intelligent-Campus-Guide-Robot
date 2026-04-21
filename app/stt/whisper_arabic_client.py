"""
Navigator - Local Arabic Whisper STT Client

Buffers raw PCM audio during a VAD turn and transcribes it locally with
faster-whisper on finalize. This is used as the Arabic side of DualSTTClient
because dedicated local Whisper handles low-quality Arabic microphone input
more reliably than cloud streaming STT.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Optional

from app.config import get_settings
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)


class WhisperArabicSTTClient:
    """
    Local faster-whisper STT client for Arabic.

    Buffers audio during VAD and transcribes on finalize.
    Dramatically better Arabic accuracy than Deepgram ar-EG on any mic.
    """

    _model = None
    _model_name: str | None = None
    _model_lock = threading.Lock()

    def __init__(
        self,
        on_partial: Optional[Callable[[TranscriptEvent], None]] = None,
        on_final: Optional[Callable[[TranscriptEvent], None]] = None,
        language: str = "ar-EG",
        keyterms: Optional[list[str]] = None,
        mock: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        del keyterms
        cfg = get_settings()
        self._language = language or "ar-EG"
        self._model_size = cfg.whisper_arabic_model
        self._mock = mock
        self._session_id = session_id
        self._connected = False
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        self._last_final_text: str | None = None

        self._on_partial = on_partial or (lambda _: None)
        self._on_final = on_final or (lambda _: None)
        self._on_connected: Callable[[], None] = lambda: None
        self._on_error: Callable[[str, str], None] = lambda *_: None

        logger.info(
            "whisper_arabic_client_init",
            model=self._model_size,
            language=self._language,
            mock=self._mock,
        )

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def set_callbacks(
        self,
        *,
        on_partial: Optional[Callable[[TranscriptEvent], None]] = None,
        on_final: Optional[Callable[[TranscriptEvent], None]] = None,
        on_connected: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if on_partial is not None:
            self._on_partial = on_partial
        if on_final is not None:
            self._on_final = on_final
        if on_connected is not None:
            self._on_connected = on_connected
        if on_error is not None:
            self._on_error = on_error

    def connect(self) -> None:
        if self._connected:
            return

        self._clear_buffer()
        self._last_final_text = None

        if self._mock:
            self._connected = True
            logger.info("whisper_mock_connected")
            self._notify_connected()
            return

        if not self._ensure_model_loaded():
            return

        self._connected = True
        logger.info("whisper_connected", language=self._language, model=self._model_size)
        self._notify_connected()

    def disconnect(self) -> None:
        self._clear_buffer()
        self._connected = False
        logger.info("whisper_disconnected")

    def send_audio(self, frame: bytes) -> None:
        if not frame:
            return
        with self._buffer_lock:
            self._audio_buffer.extend(frame)

    def finalize_turn(self) -> None:
        if self._mock:
            logger.debug("whisper_mock_finalize_skipped")
            return

        with self._buffer_lock:
            audio = bytes(self._audio_buffer)
            self._audio_buffer.clear()

        if not audio:
            logger.debug("whisper_finalize_empty_audio_skipped")
            return

        try:
            text, confidence = self._transcribe(audio)
        except Exception as exc:
            logger.error("whisper_transcribe_error", error=str(exc))
            self._notify_error("whisper_transcribe_error", str(exc))
            return

        text = text.strip()
        if not text:
            logger.debug("whisper_empty_transcript_skipped")
            return
        if text == self._last_final_text:
            logger.debug("whisper_duplicate_final_skipped", text=text[:60])
            return

        self._last_final_text = text
        event = TranscriptEvent(
            text=text,
            is_final=True,
            language=self._language,
            language_confidence=confidence,
            confidence=confidence,
            session_id=self._session_id,
            source="whisper",
        )
        logger.info("stt_final", text=text, language=event.language, confidence=event.confidence)
        self._notify_final(event)

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id

    def reset_turn(self) -> None:
        self._clear_buffer()
        self._last_final_text = None

    def inject_mock_transcript(
        self,
        text: str,
        is_final: bool = True,
        *,
        language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> None:
        if not self._mock:
            logger.warning("whisper_inject_mock_transcript_called_in_real_mode")
            return

        event = TranscriptEvent(
            text=text,
            is_final=is_final,
            language=language or self._language,
            language_confidence=language_confidence,
            confidence=0.95,
            session_id=self._session_id,
            source="whisper_mock",
        )
        if is_final:
            self._notify_final(event)
        else:
            self._notify_partial(event)

    def _ensure_model_loaded(self) -> bool:
        with self._model_lock:
            if self.__class__._model is not None and self.__class__._model_name == self._model_size:
                return True

            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:
                message = f"faster-whisper is not installed: {exc}"
                logger.error("whisper_not_installed", error=str(exc))
                self._notify_error("whisper_not_installed", message)
                return False

            try:
                self.__class__._model = WhisperModel(
                    self._model_size,
                    device="cpu",
                    compute_type="int8",
                )
                self.__class__._model_name = self._model_size
                logger.info("whisper_model_loaded", model=self._model_size, device="cpu")
                return True
            except Exception as exc:
                logger.error("whisper_model_load_error", error=str(exc), model=self._model_size)
                self._notify_error("whisper_model_load_error", str(exc))
                return False

    def _transcribe(self, audio_bytes: bytes) -> tuple[str, float]:
        """Run faster-whisper on buffered audio. Returns (text, confidence)."""
        import numpy as np

        if self.__class__._model is None and not self._ensure_model_loaded():
            return "", 0.0

        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.__class__._model.transcribe(
            pcm,
            language="ar",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, 0.85

    def _clear_buffer(self) -> None:
        with self._buffer_lock:
            self._audio_buffer.clear()

    def _notify_connected(self) -> None:
        try:
            self._on_connected()
        except Exception as exc:
            logger.error("whisper_connected_callback_error", error=str(exc))

    def _notify_partial(self, event: TranscriptEvent) -> None:
        try:
            self._on_partial(event)
        except Exception as exc:
            logger.error("whisper_partial_callback_error", error=str(exc))

    def _notify_final(self, event: TranscriptEvent) -> None:
        try:
            self._on_final(event)
        except Exception as exc:
            logger.error("whisper_final_callback_error", error=str(exc))

    def _notify_error(self, reason: str, message: str) -> None:
        try:
            self._on_error(reason, message)
        except Exception as exc:
            logger.error("whisper_error_callback_error", error=str(exc))
