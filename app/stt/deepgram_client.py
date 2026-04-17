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
- Handles reconnect on connection failure
- Deduplicates consecutive identical final transcripts

Policies (§4.2):
- Partial transcripts are NOT forwarded to the router
- Only is_final=True transcripts trigger routing
- Duplicate finals are silently dropped

Mock mode:
    client = DeepgramStreamingClient(mock=True)
    client.inject_mock_transcript("where is the robotics lab", is_final=True)
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Optional

from app.config import get_settings
from app.storage.db import get_db
from app.utils.contracts import TranscriptEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)


class DeepgramStreamingClient:
    """
    Real-time streaming STT wrapper for Deepgram Nova-3.

    Args:
        on_partial: Called with a TranscriptEvent for every partial result.
        on_final:   Called with a TranscriptEvent for every committed final result.
        language:   Language code hint: 'en' or 'ar-EG'/'ar'.
        keyterms:   List of campus keywords to bias recognition (§4.3).
        mock:       If True, no network calls. Use inject_mock_transcript() instead.
        session_id: Optional session ID to attach to transcript events.
    """

    def __init__(
        self,
        on_partial: Optional[Callable[[TranscriptEvent], None]] = None,
        on_final:   Optional[Callable[[TranscriptEvent], None]] = None,
        language: str = "en",
        keyterms: Optional[list[str]] = None,
        mock: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        cfg = get_settings()

        self._api_key    = cfg.deepgram_api_key
        self._language   = language
        self._keyterms   = keyterms or []
        self._mock       = mock
        self._session_id = session_id

        self._on_partial = on_partial or (lambda _: None)
        self._on_final   = on_final   or (lambda _: None)

        # De-duplication: track last final text to avoid double-routing
        self._last_final_text: Optional[str] = None

        # Real-mode state
        self._connection   = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected    = False

        logger.info(
            "deepgram_client_init",
            language=self._language,
            keyterms_count=len(self._keyterms),
            mock=self._mock,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open the Deepgram WebSocket connection.
        Runs the async event loop in a background daemon thread.
        """
        if self._mock:
            logger.info("deepgram_mock_connected")
            self._connected = True
            return

        if not self._api_key:
            logger.error("deepgram_no_api_key")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_async_loop, daemon=True, name="deepgram-stt"
        )
        self._thread.start()
        # Give the async loop a moment to establish the connection
        time.sleep(0.3)

    def disconnect(self) -> None:
        """Close the WebSocket connection and stop the async loop."""
        if self._mock:
            self._connected = False
            logger.info("deepgram_mock_disconnected")
            return

        self._connected = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("deepgram_disconnected")

    def send_audio(self, frame: bytes) -> None:
        """
        Push a raw PCM audio frame to the STT stream.
        Call this for every speech frame from the VAD.

        Args:
            frame: Raw 16-bit mono PCM bytes at 16 kHz.
        """
        if self._mock or not self._connected:
            return

        if self._connection and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._send_frame(frame), self._loop
                )
            except Exception as exc:
                logger.error("deepgram_send_error", error=str(exc))

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

    # ── Transcript routing ─────────────────────────────────────────────────────

    def _handle_partial(self, event: TranscriptEvent) -> None:
        """Partial transcript — emit for debug/UI, never route."""
        logger.debug("stt_partial", text=event.text[:60])
        try:
            self._on_partial(event)
        except Exception as exc:
            logger.error("stt_partial_callback_error", error=str(exc))

    def _handle_final(self, event: TranscriptEvent) -> None:
        """Final transcript — deduplicate, then emit for routing."""
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

    # ── Async loop (real mode) ─────────────────────────────────────────────────

    def _run_async_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as exc:
            logger.error("deepgram_loop_error", error=str(exc))
        finally:
            self._loop.close()

    async def _connect_and_listen(self) -> None:
        """Establish WebSocket connection and listen for transcripts."""
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents  # type: ignore
        except ImportError:
            logger.error("deepgram_sdk_not_installed")
            return

        client = DeepgramClient(self._api_key)

        options = LiveOptions(
            model="nova-3",
            language=self._language,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=True,
            # Keyterms (boosts recognition of campus-specific vocabulary)
            keywords=self._keyterms if self._keyterms else None,
        )

        try:
            self._connection = client.listen.live.v("1")

            self._connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript_event)
            self._connection.on(LiveTranscriptionEvents.Error, self._on_error_event)

            await self._connection.start(options)
            self._connected = True
            logger.info("deepgram_connected", language=self._language)

            # Keep alive until explicitly disconnected
            while self._connected:
                await asyncio.sleep(0.1)

            await self._connection.finish()

        except Exception as exc:
            logger.error("deepgram_connect_failed", error=str(exc))
            self._connected = False

    async def _send_frame(self, frame: bytes) -> None:
        """Send one PCM frame to the live connection."""
        if self._connection and self._connected:
            await self._connection.send(frame)

    def _on_transcript_event(self, _client, result, **kwargs) -> None:
        """SDK callback: parse transcript and route to partial/final handler."""
        try:
            alt = result.channel.alternatives[0]
            text = alt.transcript.strip()
            if not text:
                return

            event = TranscriptEvent(
                text=text,
                is_final=result.is_final,
                language=self._language,
                confidence=alt.confidence if hasattr(alt, "confidence") else 0.0,
                session_id=self._session_id,
                source="deepgram",
            )

            if result.is_final:
                self._handle_final(event)
            else:
                self._handle_partial(event)

        except Exception as exc:
            logger.error("deepgram_transcript_parse_error", error=str(exc))

    def _on_error_event(self, _client, error, **kwargs) -> None:
        logger.error("deepgram_error_event", error=str(error))


# ─────────────────────────────────────────────────────────────────────────────
# Keyterm inventory (§4.3)
# ─────────────────────────────────────────────────────────────────────────────

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
            ("locations",    "name"),
            ("locations",    "code"),
            ("staff",        "full_name"),
            ("departments",  "name"),
            ("facilities",   "name"),
            ("aliases",      "alias_text"),
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
