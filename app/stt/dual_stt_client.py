"""
Speculative dual-STT client for Navigator.

Runs English and Arabic Deepgram streaming clients in parallel for each
session. The first high-confidence final transcript wins, and the loser is
disconnected without blocking transcript delivery.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import replace
from typing import Callable

from app.config import get_settings
from app.stt.deepgram_client import DeepgramStreamingClient
from app.stt.whisper_arabic_client import WhisperArabicSTTClient
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_MIN_WIN_CONFIDENCE = 0.5
_TIE_WINDOW_SEC = 0.2


class DualSTTClient:
    """
    Drop-in STT wrapper that races English and Arabic Deepgram connections.

    Public methods intentionally mirror DeepgramStreamingClient so the existing
    Pipecat adapter can wrap this client without knowing it is dual-backed.
    """

    def __init__(
        self,
        *,
        on_partial: Callable[[TranscriptEvent], None] | None = None,
        on_final: Callable[[TranscriptEvent], None] | None = None,
        language: str = "en",
        keyterms: list[str] | None = None,
        mock: bool = False,
        session_id: str | None = None,
    ) -> None:
        del language, keyterms
        cfg = get_settings()
        self._en_language = "en"
        self._ar_language = cfg.deepgram_language_ar
        self._session_id = session_id

        self._en_client = DeepgramStreamingClient(
            language=self._en_language,
            mock=mock,
            session_id=session_id,
        )
        self._ar_client = WhisperArabicSTTClient(
            language=self._ar_language,
            mock=mock,
            session_id=session_id,
        )
        self._en_client._deepgram_live_language = self._en_language

        self._on_partial = on_partial
        self._on_final = on_final
        self._on_connected: Callable[[], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None

        self._lock = threading.Lock()
        self._winner_language: str | None = None
        self._winner_forwarded = False
        self._pending_candidates: dict[str, tuple[TranscriptEvent, float]] = {}
        self._tie_timer: threading.Timer | None = None

        self._bind_internal_callbacks()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def set_callbacks(
        self,
        on_partial: Callable[[TranscriptEvent], None] | None = None,
        on_final: Callable[[TranscriptEvent], None] | None = None,
        on_connected: Callable[[], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ) -> None:
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_connected = on_connected
        self._on_error = on_error
        self._bind_internal_callbacks()

    def connect(self) -> None:
        self._clear_winner_state()
        logger.info(
            "dual_stt_started",
            english_language=self._en_language,
            arabic_language=self._ar_language,
        )
        self._en_client.connect()
        self._ar_client.connect()

    def disconnect(self) -> None:
        with self._lock:
            had_winner = self._winner_forwarded
        if not had_winner:
            logger.info("dual_stt_both_failed", session_id=self._session_id)
        self._cancel_tie_timer()
        self._en_client.disconnect()
        self._ar_client.disconnect()

    def send_audio(self, frame: bytes) -> None:
        with self._lock:
            winner = self._winner_language
        if winner == self._en_language:
            self._en_client.send_audio(frame)
        elif winner == self._ar_language:
            self._ar_client.send_audio(frame)
        else:
            self._en_client.send_audio(frame)
            self._ar_client.send_audio(frame)

    def finalize_turn(self) -> None:
        self._en_client.finalize_turn()
        self._ar_client.finalize_turn()

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id
        self._en_client.set_session_id(session_id)
        self._ar_client.set_session_id(session_id)

    def reset_turn(self) -> None:
        self._en_client.reset_turn()
        self._ar_client.reset_turn()
        self._clear_winner_state()

    def inject_mock_transcript(
        self,
        text: str,
        *,
        is_final: bool = True,
        language: str | None = "en",
        language_confidence: float | None = None,
    ) -> None:
        target = self._ar_client if (language or "").lower().startswith("ar") else self._en_client
        target.inject_mock_transcript(
            text,
            is_final=is_final,
            language=language,
            language_confidence=language_confidence,
        )

    def _bind_internal_callbacks(self) -> None:
        self._en_client.set_callbacks(
            on_partial=lambda event: self._handle_partial(self._en_language, event),
            on_final=lambda event: self._handle_final(self._en_language, self._ar_language, event),
            on_connected=self._handle_connected,
            on_error=self._handle_error,
        )
        self._ar_client.set_callbacks(
            on_partial=lambda event: self._handle_partial(self._ar_language, event),
            on_final=lambda event: self._handle_final(self._ar_language, self._en_language, event),
            on_connected=self._handle_connected,
            on_error=self._handle_error,
        )

    def _handle_partial(self, language: str, event: TranscriptEvent) -> None:
        if not self._on_partial:
            return
        with self._lock:
            winner = self._winner_language
        if winner and winner != language:
            return
        self._on_partial(self._event_for_language(event, language))

    def _handle_final(
        self,
        language: str,
        loser_language: str,
        event: TranscriptEvent,
    ) -> None:
        if not event.text.strip() or event.confidence < _MIN_WIN_CONFIDENCE:
            return

        candidate = self._event_for_language(event, language)
        should_resolve = False
        with self._lock:
            if self._winner_forwarded:
                return
            self._pending_candidates[language] = (candidate, time.monotonic())
            if len(self._pending_candidates) >= 2:
                should_resolve = True
            elif self._tie_timer is None:
                self._tie_timer = threading.Timer(_TIE_WINDOW_SEC, self._resolve_pending_winner)
                self._tie_timer.daemon = True
                self._tie_timer.start()

        if should_resolve:
            self._resolve_pending_winner()

    def _resolve_pending_winner(self) -> None:
        selected_language: str | None = None
        selected_event: TranscriptEvent | None = None
        loser_language: str | None = None

        with self._lock:
            if self._winner_forwarded or not self._pending_candidates:
                return

            selected_language, selected_event = self._select_winner_locked()
            loser_language = self._other_language(selected_language)
            self._winner_language = selected_language
            self._winner_forwarded = True
            self._pending_candidates.clear()
            self._cancel_tie_timer_locked()

        logger.info(
            "dual_stt_winner",
            winning_language=selected_language,
            confidence=selected_event.confidence,
            text_preview=selected_event.text[:80],
            loser_language=loser_language,
        )
        self._disconnect_loser_async(loser_language)
        if self._on_final:
            self._on_final(selected_event)

    def _select_winner_locked(self) -> tuple[str, TranscriptEvent]:
        if len(self._pending_candidates) == 1:
            language, (event, _) = next(iter(self._pending_candidates.items()))
            return language, event

        en_event, en_time = self._pending_candidates[self._en_language]
        ar_event, ar_time = self._pending_candidates[self._ar_language]
        if abs(en_time - ar_time) <= _TIE_WINDOW_SEC:
            if ar_event.confidence > en_event.confidence:
                return self._ar_language, ar_event
            if en_event.confidence > ar_event.confidence:
                return self._en_language, en_event
            if _contains_arabic(ar_event.text) or _contains_arabic(en_event.text):
                return self._ar_language, ar_event
            return self._en_language, en_event

        if en_time <= ar_time:
            return self._en_language, en_event
        return self._ar_language, ar_event

    def _disconnect_loser_async(self, loser_language: str | None) -> None:
        if not loser_language:
            return
        loser = self._ar_client if loser_language == self._ar_language else self._en_client

        def _disconnect() -> None:
            loser.disconnect()
            logger.info("dual_stt_loser_disconnected", losing_language=loser_language)

        thread = threading.Thread(target=_disconnect, daemon=True)
        thread.start()

    def _event_for_language(self, event: TranscriptEvent, language: str) -> TranscriptEvent:
        return replace(event, language=language, source="deepgram", session_id=self._session_id)

    def _handle_connected(self) -> None:
        if self._on_connected:
            self._on_connected()

    def _handle_error(self, reason: str, message: str) -> None:
        if self._on_error:
            self._on_error(reason, message)

    def _clear_winner_state(self) -> None:
        with self._lock:
            self._winner_language = None
            self._winner_forwarded = False
            self._pending_candidates.clear()
            self._cancel_tie_timer_locked()

    def _cancel_tie_timer(self) -> None:
        with self._lock:
            self._cancel_tie_timer_locked()

    def _cancel_tie_timer_locked(self) -> None:
        if self._tie_timer is not None:
            self._tie_timer.cancel()
            self._tie_timer = None

    def _other_language(self, language: str | None) -> str | None:
        if language == self._en_language:
            return self._ar_language
        if language == self._ar_language:
            return self._en_language
        return None


def _contains_arabic(text: str) -> bool:
    return bool(_ARABIC_PATTERN.search(text))
