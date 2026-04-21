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
- Enables Deepgram automatic language detection in live mode

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
import re
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Callable
from functools import lru_cache
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
_DEEPGRAM_KEEPALIVE_SEC = 4.0
_PENDING_AUDIO_FRAMES_LIMIT = 1024
_KEYTERM_BLACKLIST: frozenset[str] = frozenset({
    "lab",
    "room",
    "office",
    "department",
    "building",
    "floor",
    "location",
    "hall",
    "area",
    "center",
    "centre",
})
_KEYTERM_MIN_LEN = 3
_KEYTERM_MAX_TOTAL = 100
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")
_ECU_STATIC_KEYTERMS: list[str] = [
    "ECU",
    "Egyptian Chinese University",
    "Faculty of Engineering",
    "Engineering Faculty",
    "Faculty of Computers",
    "Computer Science Faculty",
    "CIS Faculty",
    "Faculty of Economics",
    "Economics Faculty",
    "Faculty of Pharmacy",
    "Pharmacy Faculty",
    "Faculty of Physical Therapy",
    "PT Faculty",
    "Faculty of Arts and Design",
    "Arts Faculty",
    "Faculty of Veterinary",
    "Vet Faculty",
    "Faculty of Mass Communication",
    "Media Faculty",
    "Faculty of Law",
    "Law Faculty",
    "Student Affairs",
    "Registrar",
    "Dean's Office",
    "Research and Innovation Center",
    "R&I Center",
    "Student Portal",
    "Academic Calendar",
    "Software Engineering",
    "Systems Engineering",
    "Robotics",
    "Machine Vision",
    "Computer Networks",
    "Data Science",
    "Artificial Intelligence",
    "Doctor",
    "Professor",
    "Dr.",
    "Prof.",
]


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
        self._deepgram_live_language = language or cfg.deepgram_language or "en"
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
        self._pending_final_language: Optional[str] = None
        self._pending_final_language_confidence: Optional[float] = None
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
        self._connect_options: dict[str, Any] = {}
        self._stop_requested = threading.Event()
        self._last_audio_activity = time.monotonic()

        logger.info(
            "deepgram_client_init",
            language=self._language,
            live_language=self._deepgram_live_language,
            keyterms_count=len(self._keyterms),
            keyterm_prompting_active=bool(self._keyterms),
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
        self._connect_options = self._build_connect_options()
        keyterm_prompting_active = "keyterm" in self._connect_options

        logger.info(
            "deepgram_connect_options",
            options=self._connect_options,
            available_keyterms_count=len(self._keyterms),
            active_keyterms_count=len(self._connect_options.get("keyterm", [])),
            keyterm_prompting_active=keyterm_prompting_active,
        )

        logger.info(
            "deepgram_connecting",
            language=self._language,
            model=_DEEPGRAM_MODEL,
            keyterms_count=len(self._keyterms),
            keyterm_prompting_active=keyterm_prompting_active,
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

    def inject_mock_transcript(
        self,
        text: str,
        is_final: bool = True,
        *,
        language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> None:
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
            language=language or self._language,
            language_confidence=language_confidence,
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

    def _build_connect_options(self) -> dict[str, Any]:
        """
        Build the websocket query used for the live Nova-3 handshake.

        The connection starts from a minimal known-good option set, then
        reintroduces Nova-3 keyterm prompting using Deepgram's supported
        `keyterm` parameter when campus hints are available.
        """
        options = {
            "model": get_settings().deepgram_model or _DEEPGRAM_MODEL,
            "language": self._language,
            "encoding": _DEEPGRAM_ENCODING,
            "sample_rate": _DEEPGRAM_SAMPLE_RATE,
            "channels": _DEEPGRAM_CHANNELS,
            "interim_results": True,
            "punctuate": True,
            "smart_format": True,
            "utterance_end_ms": 1000,
            "vad_events": True,
            "endpointing": 300,
        }
        keyterm_options = self._build_nova3_keyterm_options()
        options.update(keyterm_options)
        return options

    def _build_nova3_keyterm_options(self) -> dict[str, list[str]]:
        """
        Prepare the Nova-3 keyterm option when explicitly enabled.
        """
        if not get_settings().deepgram_keyterm_prompting_enabled:
            return {}
        terms = self._keyterms[:_KEYTERM_MAX_TOTAL]
        return {"keyterm": terms} if terms else {}
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
            from deepgram.core.events import EventType  # type: ignore
            from deepgram.listen.v1.socket_client import AsyncV1SocketClient  # type: ignore
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
        connect_options = dict(self._connect_options)
        websocket = None

        try:
            ws_url, headers = self._build_websocket_request(client, connect_options)
            websocket = await self._open_websocket(ws_url, headers)
            connection = AsyncV1SocketClient(websocket=websocket)
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

        except Exception as exc:
            message = f"Deepgram websocket error: {str(exc)}"
            diagnostics = self._extract_websocket_error_details(exc)
            response_details = self._format_handshake_diagnostics(diagnostics)
            self._connect_error = message
            logger.error(
                "deepgram_connect_failed",
                error=message,
                status_code=diagnostics.get("status_code"),
                options=connect_options,
                response_details=response_details or None,
                response_probe=diagnostics or None,
            )
            self._notify_error(
                "deepgram_connect_failed",
                message if not response_details else f"{message} {response_details}",
            )
        finally:
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)
            if websocket is not None:
                try:
                    await websocket.close()
                except Exception as exc:
                    logger.warning("deepgram_websocket_close_failed", error=str(exc))

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

    def _build_websocket_request(self, client: Any, connect_options: dict[str, Any]) -> tuple[str, dict[str, str]]:
        from deepgram.core.jsonable_encoder import jsonable_encoder  # type: ignore
        from deepgram.core.query_encoder import encode_query  # type: ignore
        from deepgram.core.remove_none_from_dict import remove_none_from_dict  # type: ignore

        ws_url = client.listen.v1._raw_client._client_wrapper.get_environment().production + "/v1/listen"
        encoded_query = encode_query(jsonable_encoder(remove_none_from_dict(connect_options)))
        if encoded_query:
            normalized_query = [
                (key, str(value).lower() if isinstance(value, bool) else value)
                for key, value in encoded_query
            ]
            ws_url = ws_url + "?" + urllib.parse.urlencode(normalized_query)

        headers = client.listen.v1._raw_client._client_wrapper.get_headers()
        return ws_url, headers

    async def _open_websocket(self, ws_url: str, headers: dict[str, str]) -> Any:
        import websockets  # type: ignore

        try:
            return await websockets.connect(ws_url, additional_headers=headers)
        except TypeError:
            return await websockets.connect(ws_url, extra_headers=headers)

    def _extract_websocket_error_details(self, exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        }

        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", status_code)
            details["reason_phrase"] = getattr(response, "reason_phrase", None)

            response_headers = getattr(response, "headers", None)
            if response_headers is not None:
                details["response_headers"] = dict(response_headers)

            body = getattr(response, "body", b"")
            if body:
                if isinstance(body, (bytes, bytearray)):
                    details["response_body"] = bytes(body).decode("utf-8", errors="replace")
                else:
                    details["response_body"] = str(body)

        if status_code is not None:
            details["status_code"] = status_code

        return details

    def _format_handshake_diagnostics(self, diagnostics: dict[str, Any]) -> str:
        if not diagnostics:
            return ""

        parts: list[str] = []
        status_code = diagnostics.get("status_code")
        if status_code is not None:
            parts.append(f"status_code={status_code}")

        reason_phrase = diagnostics.get("reason_phrase")
        if reason_phrase:
            parts.append(f"reason={reason_phrase}")

        response_body = diagnostics.get("response_body")
        if response_body:
            parts.append(f"body={response_body[:240]}")

        exception_text = diagnostics.get("exception")
        if exception_text and not parts:
            parts.append(f"probe={exception_text}")

        return "; ".join(parts)

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
            detected_language = self._extract_detected_language(result, alt)
            language_confidence = self._extract_language_confidence(result, alt)

            if not is_final:
                if not text:
                    return

                event = TranscriptEvent(
                    text=text,
                    is_final=False,
                    language=detected_language,
                    language_confidence=language_confidence,
                    confidence=confidence,
                    session_id=self._session_id,
                    source="deepgram",
                )
                self._handle_partial(event)
                return

            if text:
                self._buffer_final_segment(
                    text,
                    confidence,
                    detected_language=detected_language,
                    language_confidence=language_confidence,
                )

            if speech_final or from_finalize:
                trigger = "speech_final" if speech_final else "from_finalize"
                self._flush_pending_segments(trigger=trigger)

        except Exception as exc:
            logger.error("deepgram_transcript_parse_error", error=str(exc))

    def _buffer_final_segment(
        self,
        text: str,
        confidence: float,
        *,
        detected_language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> None:
        normalized = " ".join(text.split()).strip()
        if not normalized:
            return

        if self._pending_final_segments and normalized == self._pending_final_segments[-1]:
            logger.debug("stt_final_segment_duplicate_skipped", text=normalized[:60])
            return

        self._pending_final_segments.append(normalized)
        self._pending_final_confidence = max(self._pending_final_confidence, confidence)
        if detected_language:
            self._pending_final_language = detected_language
        if language_confidence is not None:
            current = self._pending_final_language_confidence or 0.0
            self._pending_final_language_confidence = max(current, language_confidence)
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
        detected_language = self._pending_final_language or self._language
        language_confidence = self._pending_final_language_confidence
        self._clear_pending_segments()

        event = TranscriptEvent(
            text=text,
            is_final=True,
            language=detected_language,
            language_confidence=language_confidence,
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
        self._pending_final_language = None
        self._pending_final_language_confidence = None
        with self._finalize_lock:
            self._finalize_requested = False

    def _extract_detected_language(self, result: Any, alternative: Any) -> str:
        candidates = (
            getattr(result, "language", None),
            getattr(result, "detected_language", None),
            getattr(alternative, "language", None),
            getattr(alternative, "detected_language", None),
            getattr(getattr(result, "metadata", None), "language", None),
            getattr(getattr(result, "metadata", None), "detected_language", None),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return self._language

    @staticmethod
    def _extract_language_confidence(result: Any, alternative: Any) -> Optional[float]:
        candidates = (
            getattr(result, "language_confidence", None),
            getattr(alternative, "language_confidence", None),
            getattr(getattr(result, "metadata", None), "language_confidence", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return float(candidate)
            except (TypeError, ValueError):
                continue
        return None

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


def _append_keyterms(bucket: list[str], values: list[str], *, seen: set[str]) -> None:
    for raw_value in values:
        value = " ".join(str(raw_value).split()).strip()
        if not value:
            continue
        if len(value) < _KEYTERM_MIN_LEN:
            continue
        if value.casefold() in _KEYTERM_BLACKLIST:
            continue

        dedupe_key = value.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        bucket.append(value)


def _load_column_values(conn, query: str) -> list[str]:
    rows = conn.execute(query).fetchall()
    return [row[0] for row in rows if row and row[0]]


@lru_cache(maxsize=1)
def load_keyterms_from_db() -> list[str]:
    """
    Load STT keyterm hints from the SQLite truth layer.
    Returns a curated list of campus phrases to boost in Deepgram.

    Sources:
    - High-priority campus entities
    - Common aliases
    - Staff names
    """
    try:
        conn = get_db()
        groups = {
            "staff": _load_column_values(
                conn,
                "SELECT full_name FROM staff WHERE is_active=1 AND lang='en' ORDER BY full_name ASC;",
            ),
            "labs": _load_column_values(
                conn,
                "SELECT lab_name FROM labs WHERE is_active=1 AND lang='en' ORDER BY lab_name ASC;",
            ),
            "rooms": _load_column_values(
                conn,
                "SELECT room_name FROM rooms WHERE is_active=1 AND lang='en' ORDER BY room_name ASC;",
            ),
            "departments": _load_column_values(
                conn,
                "SELECT name FROM departments WHERE is_active=1 AND lang='en' ORDER BY name ASC;",
            ),
            "landmarks": _load_column_values(
                conn,
                "SELECT landmark_name FROM landmarks WHERE is_active=1 AND lang='en' ORDER BY landmark_name ASC;",
            ),
            "aliases": _load_column_values(
                conn,
                """
                SELECT alias_text FROM aliases
                WHERE lang='en' AND canonical_type IN ('room', 'lab', 'department', 'landmark', 'staff')
                ORDER BY LENGTH(alias_text) DESC, alias_text ASC;
                """,
            ),
        }

        seen_terms: set[str] = set()
        curated_terms: list[str] = []
        group_counts: dict[str, int] = {}
        for group_name in ("staff", "labs", "rooms", "departments", "landmarks", "aliases"):
            before = len(curated_terms)
            _append_keyterms(curated_terms, groups[group_name], seen=seen_terms)
            group_counts[group_name] = len(curated_terms) - before
        before_static = len(curated_terms)
        _append_keyterms(curated_terms, _ECU_STATIC_KEYTERMS, seen=seen_terms)
        group_counts["ecu_static"] = len(curated_terms) - before_static

        logger.info(
            "deepgram_keyterms_loaded",
            count=min(len(curated_terms), _KEYTERM_MAX_TOTAL),
            **{f"{name}_count": count for name, count in group_counts.items()},
        )
        return curated_terms[:_KEYTERM_MAX_TOTAL]

    except Exception as exc:
        logger.error("deepgram_keyterms_load_error", error=str(exc))
        return []


@lru_cache(maxsize=1)
def load_arabic_keyterms_from_db() -> list[str]:
    """
    Load campus terms for ElevenLabs Arabic keyword hints.
    Returns Arabic-script terms first, then English campus entity names.
    """
    try:
        conn = get_db()
        queries = [
            "SELECT full_name FROM staff WHERE is_active=1 AND lang='ar'",
            "SELECT name FROM departments WHERE is_active=1 AND lang='ar'",
            "SELECT lab_name FROM labs WHERE is_active=1 AND lang='ar'",
            "SELECT landmark_name FROM landmarks WHERE is_active=1 AND lang='ar'",
            "SELECT alias_text FROM aliases WHERE lang='ar'",
        ]
        seen: set[str] = set()
        terms: list[str] = []
        for query in queries:
            for value in _load_column_values(conn, query):
                term = " ".join(str(value).split()).strip()
                if not term or not _ARABIC_SCRIPT.search(term):
                    continue
                key = term.casefold()
                if key in seen:
                    continue
                seen.add(key)
                terms.append(term)
                if len(terms) >= _KEYTERM_MAX_TOTAL:
                    return terms

        for value in load_keyterms_from_db():
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                terms.append(value)
            if len(terms) >= _KEYTERM_MAX_TOTAL:
                break

        logger.info("elevenlabs_arabic_keyterms_loaded", count=len(terms))
        return terms[:_KEYTERM_MAX_TOTAL]
    except Exception as exc:
        logger.error("elevenlabs_arabic_keyterms_load_error", error=str(exc))
        return []
