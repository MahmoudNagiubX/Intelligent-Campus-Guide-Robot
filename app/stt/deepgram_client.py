"""
Navigator - Deepgram Streaming STT Client
Phase 4, Step 4.1 + 4.2 + 4.3

Wraps the Deepgram Python SDK for real-time speech-to-text.
Streams PCM audio frames and emits partial and final transcript events.

Behavior:
- Opens a WebSocket connection to Deepgram Nova-3
- Accepts PCM audio frames pushed via send_audio()
- Fires on_partial() for interim transcripts (display / debug only)
- Fires on_final() for committed final transcripts (routing happens here)
- Surfaces connection failures clearly in logs and callbacks
- Deduplicates consecutive identical final transcripts

Policies:
- Partial transcripts are NOT forwarded to the router
- Only committed utterance finals trigger routing
- Duplicate finals are silently dropped

Mock mode:
    client = DeepgramStreamingClient(mock=True)
    client.inject_mock_transcript("where is the robotics lab", is_final=True)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Optional

from app.config import get_settings
from app.storage.db import get_db
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_CONNECT_WAIT_SEC = 5.0
_CONNECT_POLL_SEC = 0.05
_DEEPGRAM_MODEL = "nova-3"
_DEEPGRAM_ENCODING = "linear16"
_DEEPGRAM_SAMPLE_RATE = 16000
_DEEPGRAM_CHANNELS = 1
_DEEPGRAM_ENDPOINTING_MS = 300
_DEEPGRAM_UTTERANCE_END_MS = 1000
_DEEPGRAM_KEEPALIVE_SEC = 4.0
_PENDING_AUDIO_FRAMES_LIMIT = 1024


class DeepgramStreamingClient:
    """
    Real-time streaming STT wrapper for Deepgram Nova-3.

    Args:
        on_partial: Called with a TranscriptEvent for every partial result.
        on_final:   Called with a TranscriptEvent for every committed final result.
        language:   Language code hint: "en" or "ar-EG"/"ar".
        keyterms:   List of campus keywords to bias recognition.
        mock:       If True, no network calls. Use inject_mock_transcript() instead.
        session_id: Optional session ID to attach to transcript events.
    """

    def __init__(
        self,
        on_partial: Optional[Callable[[TranscriptEvent], None]] = None,
        on_final: Optional[Callable[[TranscriptEvent], None]] = None,
        language: str = "en",
        keyterms: Optional[list[str]] = None,
        mock: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        cfg = get_settings()

        self._api_key = cfg.deepgram_api_key
        self._language = language
        self._keyterms = keyterms or []
        self._mock = mock
        self._session_id = session_id

        self._on_partial = on_partial or (lambda _: None)
        self._on_final = on_final or (lambda _: None)
        self._on_connected: Callable[[], None] = lambda: None
        self._on_error: Callable[[str, str], None] = lambda *_: None

        self._last_final_text: Optional[str] = None
        self._pending_final_segments: list[str] = []
        self._pending_final_confidence = 0.0
        self._pending_audio: deque[bytes] = deque(maxlen=_PENDING_AUDIO_FRAMES_LIMIT)
        self._pending_audio_lock = threading.Lock()
        self._finalize_requested = False
        self._finalize_lock = threading.Lock()

        self._connection = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._connect_ready = threading.Event()
        self._connect_error: Optional[str] = None
        self._stop_requested = threading.Event()
        self._last_audio_activity = time.monotonic()

        logger.info(
            "deepgram_client_init",
            language=self._language,
            keyterms_count=len(self._keyterms),
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
        """Update callback hooks without rebuilding the client."""
        if on_partial is not None:
            self._on_partial = on_partial
        if on_final is not None:
            self._on_final = on_final
        if on_connected is not None:
            self._on_connected = on_connected
        if on_error is not None:
            self._on_error = on_error

    # Public API ---------------------------------------------------------

    def connect(self) -> None:
        """
        Open the Deepgram WebSocket connection.

        Real mode runs the async websocket client inside a background thread.
        Repeated calls while already connected are ignored.
        """
        if self._mock:
            if self._connected:
                return
            self._connected = True
            logger.info("deepgram_mock_connected")
            self._notify_connected()
            return

        if self._connected or (self._thread and self._thread.is_alive()):
            return

        if not self._api_key:
            message = "DEEPGRAM_API_KEY is missing."
            logger.error("deepgram_no_api_key")
            self._notify_error("deepgram_no_api_key", message)
            return

        self._stop_requested.clear()
        self._connect_ready.clear()
        self._connect_error = None
        self._connection = None
        self._connected = False
        self._clear_pending_audio()
        self._clear_pending_segments()

        logger.info(
            "deepgram_connecting",
            language=self._language,
            model=_DEEPGRAM_MODEL,
            keyterms_count=len(self._keyterms),
        )

        self._thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True,
            name="deepgram-stt",
        )
        self._thread.start()

        if not self._connect_ready.wait(timeout=_CONNECT_WAIT_SEC):
            logger.warning("deepgram_connect_wait_timed_out", timeout_sec=_CONNECT_WAIT_SEC)

    def disconnect(self) -> None:
        """Close the WebSocket connection and stop the background loop."""
        if self._mock:
            self._connected = False
            logger.info("deepgram_mock_disconnected")
            return

        self._stop_requested.set()
        self._connect_ready.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4.0)
            if self._thread.is_alive():
                logger.warning("deepgram_disconnect_timeout")

        self._thread = None
        self._connection = None
        self._loop = None
        self._connected = False
        self._clear_pending_audio()
        self._clear_pending_segments()
        logger.info("deepgram_disconnected")

    def send_audio(self, frame: bytes) -> None:
        """
        Push a raw PCM audio frame to the STT stream.

        Args:
            frame: Raw 16-bit mono PCM bytes at 16 kHz.
        """
        if self._mock:
            return

        if not frame:
            return

        self._last_audio_activity = time.monotonic()

        if not self._connection or not self._connected:
            with self._pending_audio_lock:
                self._pending_audio.append(frame)
            return

        if self._loop is None or not self._loop.is_running():
            logger.warning("deepgram_audio_dropped_no_loop")
            return

        try:
            asyncio.run_coroutine_threadsafe(self._send_frame(frame), self._loop)
        except Exception as exc:
            logger.error("deepgram_send_schedule_error", error=str(exc))
            self._notify_error("deepgram_send_schedule_error", str(exc))

    def finalize_turn(self) -> None:
        """
        Ask Deepgram to flush all buffered audio for the current utterance.

        This is called when the local VAD reaches speech end so the turn can
        complete even if Deepgram would otherwise keep waiting for more audio.
        """
        if self._mock:
            return

        with self._finalize_lock:
            self._finalize_requested = True

        if not self._connection or not self._connected:
            logger.debug("deepgram_finalize_deferred_until_connected")
            return

        if self._loop is None or not self._loop.is_running():
            logger.warning("deepgram_finalize_skipped_no_loop")
            return

        try:
            asyncio.run_coroutine_threadsafe(self._send_finalize(), self._loop)
        except Exception as exc:
            logger.error("deepgram_finalize_schedule_error", error=str(exc))
            self._notify_error("deepgram_finalize_schedule_error", str(exc))

    def inject_mock_transcript(self, text: str, is_final: bool = True) -> None:
        """
        Inject a transcript event directly (mock mode only).
        Use this in tests to simulate STT output.
        """
        if not self._mock:
            logger.warning("inject_mock_transcript_called_in_real_mode")
            return

        event = TranscriptEvent(
            text=text,
            is_final=is_final,
            language=self._language,
            confidence=0.95,
            session_id=self._session_id,
            source="deepgram_mock",
        )

        if is_final:
            self._handle_final(event)
        else:
            self._handle_partial(event)

    def set_session_id(self, session_id: Optional[str]) -> None:
        """Update the session ID attached to future transcript events."""
        self._session_id = session_id

    def reset_turn(self) -> None:
        """Clear utterance buffers and final-transcript deduplication for a new turn."""
        self._last_final_text = None
        self._clear_pending_segments()

    # Transcript routing -------------------------------------------------

    def _handle_partial(self, event: TranscriptEvent) -> None:
        logger.debug("stt_partial", text=event.text[:60])
        try:
            self._on_partial(event)
        except Exception as exc:
            logger.error("stt_partial_callback_error", error=str(exc))

    def _handle_final(self, event: TranscriptEvent) -> None:
        text = event.text.strip()
        if not text:
            logger.debug("stt_final_empty_skipped")
            return

        if text == self._last_final_text:
            logger.debug("stt_final_duplicate_skipped", text=text[:60])
            return

        self._last_final_text = text
        logger.info("stt_final", text=text, language=event.language, confidence=event.confidence)
        try:
            self._on_final(event)
        except Exception as exc:
            logger.error("stt_final_callback_error", error=str(exc))

    # Async loop ---------------------------------------------------------

    def _run_async_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect_and_listen())
        except Exception as exc:
            logger.error("deepgram_loop_error", error=str(exc))
            self._notify_error("deepgram_loop_error", str(exc))
            self._connect_ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None
            self._connection = None
            self._connected = False
            self._connect_ready.set()

    async def _connect_and_listen(self) -> None:
        try:
            from deepgram import AsyncDeepgramClient  # type: ignore
            from deepgram.core.api_error import ApiError  # type: ignore
            from deepgram.core.events import EventType  # type: ignore
        except ImportError as exc:
            message = f"Deepgram SDK v6 imports are unavailable: {exc}"
            logger.error("deepgram_sdk_not_installed", error=str(exc))
            self._connect_error = message
            self._notify_error("deepgram_sdk_not_installed", message)
            self._connect_ready.set()
            return

        client = AsyncDeepgramClient(api_key=self._api_key)
        listener_task: Optional[asyncio.Task] = None
        keepalive_task: Optional[asyncio.Task] = None

        try:
            async with client.listen.v1.connect(
                model=_DEEPGRAM_MODEL,
                language=self._language,
                encoding=_DEEPGRAM_ENCODING,
                sample_rate=_DEEPGRAM_SAMPLE_RATE,
                channels=_DEEPGRAM_CHANNELS,
                interim_results=True,
                punctuate=True,
                smart_format=True,
                endpointing=_DEEPGRAM_ENDPOINTING_MS,
                utterance_end_ms=_DEEPGRAM_UTTERANCE_END_MS,
                vad_events=True,
                keyterm=self._keyterms if self._keyterms else None,
            ) as connection:
                self._connection = connection
                connection.on(EventType.OPEN, self._on_open_event)
                connection.on(EventType.MESSAGE, self._on_message_event)
                connection.on(EventType.ERROR, self._on_error_event)
                connection.on(EventType.CLOSE, self._on_close_event)

                listener_task = asyncio.create_task(connection.start_listening(), name="deepgram-listener")
                keepalive_task = asyncio.create_task(self._keepalive_loop(), name="deepgram-keepalive")

                while not self._stop_requested.is_set():
                    if listener_task.done():
                        await self._raise_listener_error(listener_task)
                        break
                    await asyncio.sleep(_CONNECT_POLL_SEC)

                if self._stop_requested.is_set() and self._connection is not None:
                    await self._finalize_if_pending(trigger="disconnect")
                    try:
                        await self._connection.send_close_stream()
                    except Exception as exc:
                        logger.warning("deepgram_close_stream_failed", error=str(exc))

                if listener_task is not None:
                    try:
                        await asyncio.wait_for(listener_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        listener_task.cancel()
                        await asyncio.gather(listener_task, return_exceptions=True)

        except ApiError as exc:  # pragma: no cover - exercised only with live credentials
            message = f"Deepgram API error: {getattr(exc, 'body', str(exc))}"
            self._connect_error = message
            logger.error(
                "deepgram_connect_failed",
                error=message,
                status_code=getattr(exc, "status_code", None),
            )
            self._notify_error("deepgram_connect_failed", message)
        except Exception as exc:
            self._connect_error = str(exc)
            logger.error("deepgram_connect_failed", error=str(exc))
            self._notify_error("deepgram_connect_failed", str(exc))
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)

    async def _raise_listener_error(self, listener_task: asyncio.Task) -> None:
        try:
            await listener_task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Deepgram listener failed: {exc}") from exc

    async def _send_frame(self, frame: bytes) -> None:
        if not self._connection or not self._connected:
            return

        try:
            await self._connection.send_media(frame)
        except Exception as exc:
            logger.error("deepgram_send_error", error=str(exc))
            self._notify_error("deepgram_send_error", str(exc))

    async def _send_finalize(self) -> None:
        if not self._connection or not self._connected:
            return

        try:
            await self._connection.send_finalize()
            logger.info("deepgram_finalize_requested")
        except Exception as exc:
            logger.error("deepgram_finalize_failed", error=str(exc))
            self._notify_error("deepgram_finalize_failed", str(exc))

    async def _finalize_if_pending(self, *, trigger: str) -> None:
        should_finalize = False
        with self._finalize_lock:
            should_finalize = self._finalize_requested
            self._finalize_requested = False

        if should_finalize:
            logger.debug("deepgram_finalize_flush", trigger=trigger)
            await self._send_finalize()

    async def _keepalive_loop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(_DEEPGRAM_KEEPALIVE_SEC)
            if not self._connected or not self._connection:
                continue
            if time.monotonic() - self._last_audio_activity < _DEEPGRAM_KEEPALIVE_SEC:
                continue

            try:
                await self._connection.send_keep_alive()
                logger.debug("deepgram_keepalive_sent")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("deepgram_keepalive_failed", error=str(exc))
                self._notify_error("deepgram_keepalive_failed", str(exc))
                return

    # Deepgram websocket callbacks --------------------------------------

    def _on_open_event(self, _event: object) -> None:
        self._connected = True
        self._last_audio_activity = time.monotonic()
        self._connect_ready.set()
        logger.info("deepgram_connected", language=self._language, model=_DEEPGRAM_MODEL)
        self._notify_connected()

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._flush_pending_audio_and_finalize_if_needed(),
                self._loop,
            )

    def _on_message_event(self, message: object) -> None:
        self._handle_deepgram_message(message)

    def _on_error_event(self, error: object) -> None:
        message = str(error)
        if not self._connect_ready.is_set():
            self._connect_error = message
            self._connect_ready.set()
        logger.error("deepgram_error_event", error=message)
        self._notify_error("deepgram_error_event", message)

    def _on_close_event(self, _event: object) -> None:
        was_connected = self._connected
        self._connected = False
        self._connect_ready.set()

        if self._stop_requested.is_set():
            logger.info("deepgram_connection_closed")
            return

        if was_connected:
            message = "Deepgram websocket closed unexpectedly."
            logger.warning("deepgram_connection_closed_unexpectedly")
            self._notify_error("deepgram_connection_closed_unexpectedly", message)

    async def _flush_pending_audio(self) -> None:
        while True:
            with self._pending_audio_lock:
                if not self._pending_audio:
                    break
                frame = self._pending_audio.popleft()
            await self._send_frame(frame)

    async def _flush_pending_audio_and_finalize_if_needed(self) -> None:
        await self._flush_pending_audio()
        await self._finalize_if_pending(trigger="open")

    def _handle_deepgram_message(self, message: Any) -> None:
        message_type = str(getattr(message, "type", "") or "")

        if message_type == "Results":
            self._handle_results_message(message)
            return

        if message_type == "UtteranceEnd":
            logger.debug("deepgram_utterance_end")
            self._flush_pending_segments(trigger="utterance_end")
            return

        if message_type == "SpeechStarted":
            logger.debug("deepgram_speech_started")
            return

        if message_type == "Metadata":
            logger.debug("deepgram_metadata_received")
            return

    def _handle_results_message(self, result: Any) -> None:
        try:
            alternatives = getattr(getattr(result, "channel", None), "alternatives", []) or []
            if not alternatives:
                return

            alt = alternatives[0]
            text = str(getattr(alt, "transcript", "") or "").strip()
            confidence = float(getattr(alt, "confidence", 0.0) or 0.0)
            is_final = bool(getattr(result, "is_final", False))
            speech_final = bool(getattr(result, "speech_final", False))
            from_finalize = bool(getattr(result, "from_finalize", False))

            if not is_final:
                if not text:
                    return

                event = TranscriptEvent(
                    text=text,
                    is_final=False,
                    language=self._language,
                    confidence=confidence,
                    session_id=self._session_id,
                    source="deepgram",
                )
                self._handle_partial(event)
                return

            if text:
                self._buffer_final_segment(text, confidence)

            if speech_final or from_finalize:
                trigger = "speech_final" if speech_final else "from_finalize"
                self._flush_pending_segments(trigger=trigger)

        except Exception as exc:
            logger.error("deepgram_transcript_parse_error", error=str(exc))

    def _buffer_final_segment(self, text: str, confidence: float) -> None:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return

        if self._pending_final_segments and normalized == self._pending_final_segments[-1]:
            logger.debug("stt_final_segment_duplicate_skipped", text=normalized[:60])
            return

        self._pending_final_segments.append(normalized)
        self._pending_final_confidence = max(self._pending_final_confidence, confidence)
        logger.debug(
            "stt_final_segment_buffered",
            count=len(self._pending_final_segments),
            text=normalized[:60],
        )

    def _flush_pending_segments(self, *, trigger: str) -> None:
        if not self._pending_final_segments:
            return

        text = " ".join(self._pending_final_segments).strip()
        confidence = self._pending_final_confidence
        self._clear_pending_segments()

        event = TranscriptEvent(
            text=text,
            is_final=True,
            language=self._language,
            confidence=confidence,
            session_id=self._session_id,
            source="deepgram",
        )
        logger.debug("stt_final_flush", trigger=trigger, text=text[:80])
        self._handle_final(event)

    def _clear_pending_audio(self) -> None:
        with self._pending_audio_lock:
            self._pending_audio.clear()

    def _clear_pending_segments(self) -> None:
        self._pending_final_segments.clear()
        self._pending_final_confidence = 0.0
        with self._finalize_lock:
            self._finalize_requested = False

    def _notify_connected(self) -> None:
        try:
            self._on_connected()
        except Exception as exc:
            logger.error("deepgram_connected_callback_error", error=str(exc))

    def _notify_error(self, reason: str, message: str) -> None:
        try:
            self._on_error(reason, message)
        except Exception as exc:
            logger.error("deepgram_error_callback_error", error=str(exc))


def load_keyterms_from_db() -> list[str]:
    """
    Load STT keyterm hints from the SQLite truth layer.
    Returns a flat list of campus vocabulary words to boost in Deepgram.

    Sources:
    - Location names and codes
    - Staff names
    - Department names
    - Facility names
    - All alias texts
    """
    try:
        conn = get_db()
        terms: list[str] = []

        for table, col in [
            ("locations", "name"),
            ("locations", "code"),
            ("staff", "full_name"),
            ("departments", "name"),
            ("facilities", "name"),
            ("aliases", "alias_text"),
        ]:
            rows = conn.execute(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL;").fetchall()
            for row in rows:
                val = row[0].strip()
                if val:
                    terms.append(val)

        logger.info("deepgram_keyterms_loaded", count=len(terms))
        return terms

    except Exception as exc:
        logger.error("deepgram_keyterms_load_error", error=str(exc))
        return []
