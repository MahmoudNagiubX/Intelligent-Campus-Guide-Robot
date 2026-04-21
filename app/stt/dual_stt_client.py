"""
Dual-STT client for Navigator.

Deepgram is the English specialist. ElevenLabs is the Arabic specialist. The
client arbitrates finals, holds likely phonetic-Arabic Deepgram results for the
Arabic leg, and suppresses duplicate final emissions per session.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Callable, Optional

from app.config import get_settings
from app.stt.deepgram_client import DeepgramStreamingClient, load_arabic_keyterms_from_db
from app.stt.elevenlabs_arabic_client import ElevenLabsArabicClient
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
_MIN_WIN_CONFIDENCE = 0.5
_DEDUP_WINDOW = 3
_DEDUP_MAX_RATIO = 0.88
_PHONETIC_ARABIC_WORDS: frozenset[str] = frozenset(
    {
        "wayn",
        "ayn",
        "fein",
        "fin",
        "fenin",
        "winein",
        "maktab",
        "maktaba",
        "qism",
        "kulliya",
        "mabna",
        "dor",
        "tabet",
        "tani",
        "talet",
        "modarraj",
        "madraj",
        "qa3a",
        "doktor",
        "duktur",
        "ostaz",
        "ostaza",
        "dektura",
        "khodni",
        "wadini",
        "rohni",
        "mashy",
        "3ayiz",
        "3ayz",
        "3awiz",
        "fe",
        "fi",
        "leh",
        "mesh",
        "mish",
        "ana",
        "enta",
        "hena",
        "hnak",
        "mn",
        "3n",
        "fy",
    }
)
_PHONETIC_ARABIC_MIN_RATIO: float = 0.30


class DualSTTClient:
    """Drop-in STT wrapper that races Deepgram English and ElevenLabs Arabic."""

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
        del language
        self._en_language = "en"
        self._ar_language = "ar-EG"
        self._keyterms = keyterms or []
        self._session_id = session_id

        self._deepgram_client = DeepgramStreamingClient(
            language=self._en_language,
            keyterms=self._keyterms,
            mock=mock,
            session_id=session_id,
        )
        arabic_keyterms = load_arabic_keyterms_from_db() if keyterms is None else keyterms
        self._arabic_client = ElevenLabsArabicClient(
            language=self._ar_language,
            keyterms=arabic_keyterms,
            mock=mock,
            session_id=session_id,
        )
        # Backward-compatible internal aliases used by existing tests.
        self._en_client = self._deepgram_client
        self._ar_client = self._arabic_client

        self._on_partial = on_partial
        self._on_final = on_final
        self._on_connected: Callable[[], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None

        self._lock = threading.Lock()
        self._winner_language: str | None = None
        self._winner_forwarded = False
        self._recent_emissions: list[str] = []

        self._held_deepgram_event: Optional[TranscriptEvent] = None
        self._hold_deadline: Optional[float] = None
        self._hold_timer: Optional[threading.Timer] = None
        self._hold_lock = threading.Lock()

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
        logger.info("dual_stt_started", english_language=self._en_language, arabic_language=self._ar_language)
        self._deepgram_client.connect()
        self._arabic_client.connect()
        logger.info("dual_stt.arabic_connected_eagerly", session_id=self._session_id)

    def disconnect(self) -> None:
        self._clear_hold()
        self._deepgram_client.disconnect()
        self._arabic_client.disconnect()

    def send_audio(self, frame: bytes) -> None:
        with self._lock:
            winner = self._winner_language
        if winner == self._en_language:
            self._deepgram_client.send_audio(frame)
        elif winner == self._ar_language:
            self._arabic_client.send_audio(frame)
        else:
            self._deepgram_client.send_audio(frame)
            self._arabic_client.send_audio(frame)

    def finalize_turn(self) -> None:
        self._deepgram_client.finalize_turn()
        self._arabic_client.finalize_turn()

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id
        self._deepgram_client.set_session_id(session_id)
        self._arabic_client.set_session_id(session_id)

    def reset_turn(self) -> None:
        self._deepgram_client.reset_turn()
        self._arabic_client.reset_turn()
        self._recent_emissions.clear()
        self._clear_winner_state()

    def inject_mock_transcript(
        self,
        text: str,
        *,
        is_final: bool = True,
        language: str | None = "en",
        language_confidence: float | None = None,
    ) -> None:
        target = self._arabic_client if (language or "").lower().startswith("ar") else self._deepgram_client
        target.inject_mock_transcript(
            text,
            is_final=is_final,
            language=language,
            language_confidence=language_confidence,
        )

    def _bind_internal_callbacks(self) -> None:
        self._deepgram_client.set_callbacks(
            on_partial=self._handle_deepgram_partial,
            on_final=self._on_deepgram_final,
            on_connected=self._handle_connected,
            on_error=self._handle_error,
        )
        self._arabic_client.set_callbacks(
            on_partial=self._handle_arabic_partial,
            on_final=self._on_arabic_final,
            on_connected=self._handle_connected,
            on_error=self._handle_error,
        )

    def _handle_deepgram_partial(self, event: TranscriptEvent) -> None:
        self._handle_partial(self._en_language, event)

    def _handle_arabic_partial(self, event: TranscriptEvent) -> None:
        self._handle_partial(self._ar_language, event)

    def _handle_partial(self, language: str, event: TranscriptEvent) -> None:
        if not self._on_partial:
            return
        with self._lock:
            winner = self._winner_language
        if winner and winner != language:
            return
        self._on_partial(self._event_for_language(event, language))

    def _on_deepgram_final(self, event: TranscriptEvent) -> None:
        if not event.text.strip() or event.confidence < _MIN_WIN_CONFIDENCE:
            return
        candidate = self._event_for_language(event, self._en_language)
        if _looks_like_phonetic_arabic(candidate.text):
            cfg = get_settings()
            hold_sec = cfg.stt_arabic_hold_max_ms / 1000.0
            logger.info(
                "dual_stt.phonetic_hold_extended",
                text=candidate.text[:60],
                hold_max_ms=cfg.stt_arabic_hold_max_ms,
                session_id=self._session_id,
            )
            with self._hold_lock:
                self._held_deepgram_event = candidate
                self._hold_deadline = time.monotonic() + hold_sec
                if self._hold_timer is not None:
                    self._hold_timer.cancel()
                self._hold_timer = threading.Timer(hold_sec, self._check_hold_expiry)
                self._hold_timer.daemon = True
                self._hold_timer.start()
            return
        self._try_emit_deepgram(candidate)

    def _on_arabic_final(self, event: TranscriptEvent) -> None:
        if not event.text.strip() or event.confidence < _MIN_WIN_CONFIDENCE:
            return
        candidate = self._event_for_language(event, self._ar_language)
        with self._hold_lock:
            held = self._held_deepgram_event
            self._held_deepgram_event = None
            self._hold_deadline = None
            if self._hold_timer is not None:
                self._hold_timer.cancel()
                self._hold_timer = None
            if held is not None:
                logger.info(
                    "dual_stt.phonetic_hold_resolved",
                    arabic_text=candidate.text[:60],
                    session_id=self._session_id,
                )
        self._emit_final(candidate, self._ar_language)

    def _try_emit_deepgram(self, event: TranscriptEvent) -> None:
        self._emit_final(event, self._en_language)

    def _check_hold_expiry(self) -> None:
        with self._hold_lock:
            self._hold_timer = None
            if self._held_deepgram_event is None:
                return
            event = self._held_deepgram_event
            self._held_deepgram_event = None
            self._hold_deadline = None

        logger.warning(
            "dual_stt.phonetic_hold_expired",
            text=event.text[:60],
            detail="ElevenLabs did not fire within hold window; emitting Deepgram result",
            session_id=self._session_id,
        )
        self._try_emit_deepgram(event)

    def _emit_final(self, event: TranscriptEvent, language: str) -> None:
        with self._lock:
            if self._winner_forwarded:
                return
            if self._is_duplicate_emission(event.text):
                logger.debug("dual_stt.duplicate_emission_skipped", text=event.text[:60], session_id=self._session_id)
                return
            self._winner_language = language
            self._winner_forwarded = True
            self._record_emission(event.text)

        logger.info(
            "dual_stt_winner",
            winning_language=language,
            confidence=event.confidence,
            text_preview=event.text[:80],
            session_id=self._session_id,
        )
        self._disconnect_loser_async(self._other_language(language))
        if self._on_final:
            self._on_final(event)

    def _is_duplicate_emission(self, text: str) -> bool:
        """Return True if text is too similar to a recently emitted transcript."""
        for previous in self._recent_emissions:
            ratio = SequenceMatcher(None, text.lower(), previous.lower()).ratio()
            if ratio >= _DEDUP_MAX_RATIO:
                return True
        return False

    def _record_emission(self, text: str) -> None:
        self._recent_emissions.append(text)
        if len(self._recent_emissions) > _DEDUP_WINDOW:
            self._recent_emissions.pop(0)

    def _disconnect_loser_async(self, loser_language: str | None) -> None:
        if not loser_language:
            return
        loser = self._arabic_client if loser_language == self._ar_language else self._deepgram_client

        def _disconnect() -> None:
            loser.disconnect()
            logger.info("dual_stt_loser_disconnected", losing_language=loser_language)

        threading.Thread(target=_disconnect, daemon=True).start()

    def _event_for_language(self, event: TranscriptEvent, language: str) -> TranscriptEvent:
        source = "elevenlabs" if language == self._ar_language else "deepgram"
        lang_code = "ar-EG" if language == self._ar_language else "en"
        return replace(event, language=lang_code, source=source, session_id=self._session_id)

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
        self._clear_hold()

    def _clear_hold(self) -> None:
        with self._hold_lock:
            if self._hold_timer is not None:
                self._hold_timer.cancel()
                self._hold_timer = None
            self._held_deepgram_event = None
            self._hold_deadline = None

    def _other_language(self, language: str | None) -> str | None:
        if language == self._en_language:
            return self._ar_language
        if language == self._ar_language:
            return self._en_language
        return None


def _contains_arabic(text: str) -> bool:
    return bool(_ARABIC_PATTERN.search(text))


def _looks_like_phonetic_arabic(text: str) -> bool:
    """
    Return True if Latin-script text looks like phonetic Arabic transliteration.
    """
    words = _WORD_PATTERN.findall(text.lower())
    if not words:
        return False
    matches = sum(1 for word in words if word in _PHONETIC_ARABIC_WORDS)
    return (matches / len(words)) >= _PHONETIC_ARABIC_MIN_RATIO
