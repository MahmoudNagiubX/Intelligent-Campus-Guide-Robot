"""
Navigator - ElevenLabs Arabic STT Client

ElevenLabs Scribe v2 Realtime Arabic-specialist STT client.

Drop-in replacement for WhisperArabicSTTClient inside dual_stt_client.py.
The external interface matches DeepgramStreamingClient closely:
  - connect() / disconnect()
  - send_audio(frame: bytes)
  - finalize_turn()
  - inject_mock_transcript(text, is_final, *, language, language_confidence)
  - set_callbacks(on_partial, on_final, on_connected, on_error)
  - set_session_id(session_id)
  - reset_turn()

Arabic-only filter:
  This client only forwards Arabic results (ar, ar-EG, ar-SA). If
  ElevenLabs returns English, Deepgram's English specialist should win.
"""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.parse
from collections import deque
from collections.abc import Callable
from typing import Optional

from app.config import get_settings
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)

_WS_BASE = "wss://api.elevenlabs.io/v1/speech-to-text/stream"
_CONNECT_WAIT_SEC = 6.0
_KEEPALIVE_SEC = 5.0
_PENDING_AUDIO_LIMIT = 1024
_ARABIC_LANG_PREFIXES = ("ar",)
_ELEVENLABS_PERMANENTLY_DISABLED: bool = False


class ElevenLabsArabicClient:
    """
    Real-time streaming STT wrapper for ElevenLabs Scribe v2 Realtime.

    Args:
        on_partial: Called with a TranscriptEvent for Arabic partial results.
        on_final: Called with a TranscriptEvent for Arabic final results.
        language: Arabic language hint attached to mock/default events.
        keyterms: Campus keywords passed as recognition hints.
        mock: If True, no network calls. Use inject_mock_transcript().
        session_id: Optional session ID attached to transcript events.
    """

    _process_permanently_disabled = False

    def __init__(
        self,
        on_partial: Optional[Callable[[TranscriptEvent], None]] = None,
        on_final: Optional[Callable[[TranscriptEvent], None]] = None,
        language: str = "ar-EG",
        keyterms: Optional[list[str]] = None,
        mock: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        cfg = get_settings()
        self._api_key = cfg.elevenlabs_api_key
        self._model = cfg.elevenlabs_model
        self._language = language or "ar-EG"
        self._keyterms = (keyterms or [])[: cfg.elevenlabs_keyterms_max]
        self._partial_debug = cfg.elevenlabs_partial_debug
        self._mock = mock
        self._session_id = session_id

        self._on_partial = on_partial or (lambda _: None)
        self._on_final = on_final or (lambda _: None)
        self._on_connected: Callable[[], None] = lambda: None
        self._on_error: Callable[[str, str], None] = lambda *_: None

        self._pending_audio: deque[bytes] = deque(maxlen=_PENDING_AUDIO_LIMIT)
        self._pending_audio_lock = threading.Lock()

        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._connect_ready = threading.Event()
        self._connect_error: Optional[str] = None
        self._stop_requested = threading.Event()
        self._last_final_text: Optional[str] = None
        self._permanently_disabled = self.__class__._process_permanently_disabled

        logger.info(
            "elevenlabs_arabic_client_init",
            model=self._model,
            keyterms_count=len(self._keyterms),
            mock=self._mock,
            session_id=self._session_id,
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
        """Open the ElevenLabs WebSocket. Blocks until connected or timeout."""
        if get_settings().english_only_mode:
            logger.info("elevenlabs_stt.skipped_english_only_mode")
            self._connected = True
            self._connect_ready.set()
            return

        if self._mock:
            if self._connected:
                return
            self._connected = True
            logger.info("elevenlabs_arabic_mock_connected")
            self._notify_connected()
            return

        global _ELEVENLABS_PERMANENTLY_DISABLED

        from app.stt.dual_stt_client import _elevenlabs_permanently_disabled

        if _ELEVENLABS_PERMANENTLY_DISABLED or _elevenlabs_permanently_disabled:
            logger.debug("elevenlabs_stt.skipped_permanently_disabled")
            self._permanently_disabled = True
            self._connect_ready.set()
            return

        if self._permanently_disabled or self.__class__._process_permanently_disabled:
            self._permanently_disabled = True
            logger.info("elevenlabs_arabic_permanently_disabled", session_id=self._session_id)
            self._connect_ready.set()
            return

        if self._connected or (self._thread and self._thread.is_alive()):
            return

        if not self._api_key:
            message = "ELEVENLABS_API_KEY is missing."
            logger.error("elevenlabs_arabic_no_api_key")
            self._notify_error("elevenlabs_arabic_no_api_key", message)
            return

        self._stop_requested.clear()
        self._connect_ready.clear()
        self._connect_error = None
        self._connected = False
        self._clear_pending_audio()

        self._thread = threading.Thread(
            target=self._run_async_loop,
            daemon=True,
            name="elevenlabs-arabic-stt",
        )
        self._thread.start()

        if not self._connect_ready.wait(timeout=_CONNECT_WAIT_SEC):
            logger.warning("elevenlabs_arabic_connect_timeout", timeout_sec=_CONNECT_WAIT_SEC)

    def permanently_disable(self) -> None:
        """Disable ElevenLabs Arabic reconnect attempts for this process."""
        global _ELEVENLABS_PERMANENTLY_DISABLED
        _ELEVENLABS_PERMANENTLY_DISABLED = True
        self.__class__._process_permanently_disabled = True
        self._permanently_disabled = True
        self.disconnect()

    def disconnect(self) -> None:
        """Close the WebSocket and stop the background thread."""
        if self._mock:
            self._connected = False
            logger.info("elevenlabs_arabic_mock_disconnected")
            return

        self._stop_requested.set()
        self._connect_ready.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4.0)
            if self._thread.is_alive():
                logger.warning("elevenlabs_arabic_disconnect_timeout")

        self._thread = None
        self._ws = None
        self._loop = None
        self._connected = False
        self._clear_pending_audio()
        logger.info("elevenlabs_arabic_disconnected")

    def send_audio(self, frame: bytes) -> None:
        """Push a raw PCM frame, linear16 16 kHz mono, to the stream."""
        if self._mock or not frame:
            return

        if not self._connected:
            with self._pending_audio_lock:
                self._pending_audio.append(frame)
            return

        if self._loop is None or not self._loop.is_running():
            logger.warning("elevenlabs_arabic_audio_dropped_no_loop")
            return

        try:
            asyncio.run_coroutine_threadsafe(self._send_frame(frame), self._loop)
        except Exception as exc:
            logger.error("elevenlabs_arabic_send_schedule_error", error=str(exc))
            self._notify_error("elevenlabs_arabic_send_schedule_error", str(exc))

    def finalize_turn(self) -> None:
        """
        Keep interface parity with DeepgramStreamingClient.

        ElevenLabs Scribe emits finals from the stream itself, so there is no
        explicit finalize command to send here.
        """
        logger.debug("elevenlabs_arabic_finalize_noop")

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._session_id = session_id

    def reset_turn(self) -> None:
        self._last_final_text = None

    def inject_mock_transcript(
        self,
        text: str,
        is_final: bool = True,
        *,
        language: Optional[str] = None,
        language_confidence: Optional[float] = None,
    ) -> None:
        """Inject a transcript event directly, mock mode only."""
        if not self._mock:
            logger.warning("elevenlabs_inject_called_in_real_mode")
            return

        lang = language or self._language
        event = TranscriptEvent(
            text=text,
            is_final=is_final,
            language=_normalise_lang(lang),
            language_confidence=language_confidence,
            confidence=0.95,
            session_id=self._session_id,
            source="elevenlabs_mock",
        )
        if is_final:
            self._handle_final(event)
        else:
            self._handle_partial(event)

    def _run_async_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connect_and_listen())
        except Exception as exc:
            logger.error("elevenlabs_arabic_loop_error", error=str(exc))
            self._connect_error = str(exc)
            self._notify_error("elevenlabs_arabic_loop_error", str(exc))
            self._connect_ready.set()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None
            self._ws = None
            self._connected = False
            self._connect_ready.set()

    async def _connect_and_listen(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            message = f"websockets is not installed: {exc}"
            logger.error("elevenlabs_websockets_not_installed", error=str(exc))
            self._notify_error("elevenlabs_websockets_not_installed", message)
            self._connect_ready.set()
            return

        url = self._build_ws_url()
        try:
            async with websockets.connect(
                url,
                ping_interval=_KEEPALIVE_SEC,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                self._connected = True
                self._connect_ready.set()
                logger.info("elevenlabs_arabic_connected", session_id=self._session_id)
                self._notify_connected()
                await self._flush_pending_audio()

                async for raw in ws:
                    if self._stop_requested.is_set():
                        break
                    self._parse_and_dispatch(raw)

        except Exception as exc:
            logger.warning("elevenlabs_arabic_ws_error", error=str(exc), session_id=self._session_id)
            self._connect_error = str(exc)
            self._notify_error("elevenlabs_arabic_ws_error", str(exc))
        finally:
            self._connect_ready.set()

    async def _flush_pending_audio(self) -> None:
        with self._pending_audio_lock:
            frames = list(self._pending_audio)
            self._pending_audio.clear()
        for frame in frames:
            await self._send_frame(frame)

    async def _send_frame(self, frame: bytes) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(frame)
        except Exception as exc:
            logger.debug("elevenlabs_arabic_send_error", error=str(exc))

    def _parse_and_dispatch(self, raw: str | bytes) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("elevenlabs_arabic_unparseable", preview=str(raw)[:80])
            return

        msg_type = data.get("type", "")

        if msg_type == "error":
            err = data.get("error", {})
            logger.error("elevenlabs_arabic_api_error", code=err.get("code"), message=err.get("message"))
            self._notify_error("elevenlabs_api_error", str(err))
            return

        if msg_type != "transcript":
            logger.debug("elevenlabs_arabic_unknown_msg_type", msg_type=msg_type)
            return

        transcript = data.get("transcript", {})
        text = (transcript.get("text") or "").strip()
        is_final = transcript.get("type") == "final"
        lang_code = transcript.get("language_code") or "unknown"
        confidence = float(transcript.get("confidence") or 1.0)

        if not text:
            return

        if not _is_arabic_lang(lang_code):
            logger.debug(
                "elevenlabs_arabic_non_arabic_dropped",
                lang_code=lang_code,
                text_preview=text[:40],
            )
            return

        event = TranscriptEvent(
            text=text,
            is_final=is_final,
            language=_normalise_lang(lang_code),
            language_confidence=confidence,
            confidence=confidence,
            session_id=self._session_id,
            source="elevenlabs",
        )

        if is_final:
            self._handle_final(event)
        else:
            self._handle_partial(event)

    def _handle_final(self, event: TranscriptEvent) -> None:
        text = (event.text or "").strip()
        if not text:
            return
        if text == self._last_final_text:
            logger.debug("elevenlabs_arabic_dedup_skipped", text=text[:60], session_id=self._session_id)
            return

        self._last_final_text = text
        logger.info(
            "elevenlabs_arabic_final",
            text=text[:80],
            language=event.language,
            confidence=event.confidence,
            session_id=self._session_id,
        )
        self._notify_final(event)

    def _handle_partial(self, event: TranscriptEvent) -> None:
        if self._partial_debug:
            logger.debug("elevenlabs_arabic_partial", text_preview=(event.text or "")[:40])
        self._notify_partial(event)

    def _build_ws_url(self) -> str:
        params: dict[str, str] = {
            "model_id": self._model,
            "language_code": "auto",
            "xi" + "-api-key": self._api_key,
        }
        if self._keyterms:
            params["keywords"] = ",".join(self._keyterms)
        return f"{_WS_BASE}?" + urllib.parse.urlencode(params)

    def _clear_pending_audio(self) -> None:
        with self._pending_audio_lock:
            self._pending_audio.clear()

    def _notify_connected(self) -> None:
        try:
            self._on_connected()
        except Exception as exc:
            logger.error("elevenlabs_connected_callback_error", error=str(exc))

    def _notify_final(self, event: TranscriptEvent) -> None:
        try:
            self._on_final(event)
        except Exception as exc:
            logger.error("elevenlabs_final_callback_error", error=str(exc))

    def _notify_partial(self, event: TranscriptEvent) -> None:
        try:
            self._on_partial(event)
        except Exception as exc:
            logger.error("elevenlabs_partial_callback_error", error=str(exc))

    def _notify_error(self, reason: str, message: str) -> None:
        try:
            self._on_error(reason, message)
        except Exception as exc:
            logger.error("elevenlabs_error_callback_error", error=str(exc))


def _is_arabic_lang(code: str) -> bool:
    """Return True for any Arabic BCP-47 tag, such as ar, ar-EG, or ar-SA."""
    return code.lower().startswith(_ARABIC_LANG_PREFIXES)


def _normalise_lang(code: str) -> str:
    normalized = code.strip()
    lower = normalized.lower()
    if "eg" in lower:
        return "ar-EG"
    if lower.startswith("ar"):
        return "ar"
    if lower.startswith("en"):
        return "en"
    return normalized
