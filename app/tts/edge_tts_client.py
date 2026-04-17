"""
Navigator - edge-tts TTS Client
Phase 7, Step 7.1

Wraps the edge-tts library for async text-to-speech synthesis.
Selects the correct neural voice based on the response language.

Supported voices (locked MVP):
- English: en-US-JennyNeural
- Arabic:  ar-EG-SalmaNeural

Usage:
    tts = EdgeTTSClient()
    audio_bytes = asyncio.run(tts.synthesize("Hello!", "en"))

Mock mode:
    tts = EdgeTTSClient(mock=True)
    audio_bytes = asyncio.run(tts.synthesize("Hello!", "en"))
    # Returns silent WAV bytes, no network call
"""

from __future__ import annotations

import asyncio
import io
import struct
import wave
from typing import Optional

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Silent WAV placeholder (1 second of silence at 22050 Hz, 16-bit, mono)
_SILENCE_DURATION_SAMPLES = 22050
_SILENT_WAV: bytes = b""


def _build_silent_wav(duration_ms: int = 300) -> bytes:
    """Build a valid silent WAV bytes object for use in mock mode."""
    sample_rate = 22050
    n_samples = int(sample_rate * duration_ms / 1000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(n_samples * 2))  # silence
    return buf.getvalue()


class EdgeTTSClient:
    """
    Text-to-speech client using Microsoft edge-tts.

    Args:
        mock: If True, return synthetic silence bytes instead of calling
              the edge-tts service (for testing without network).
    """

    def __init__(self, mock: bool = False) -> None:
        cfg = get_settings()
        self._voice_en = cfg.edge_tts_voice_en    # "en-US-JennyNeural"
        self._voice_ar = cfg.edge_tts_voice_ar    # "ar-EG-SalmaNeural"
        self._mock     = mock

        logger.info(
            "tts_client_init",
            voice_en=self._voice_en,
            voice_ar=self._voice_ar,
            mock=self._mock,
        )

    def voice_for(self, language: str) -> str:
        """Return the locked neural voice name for the given language code."""
        if language in ("ar", "ar-EG", "ar-SA"):
            return self._voice_ar
        return self._voice_en

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Synthesize text to speech and return raw audio bytes (MP3 in real mode).

        Args:
            text:     The text to speak. Must be non-empty.
            language: Language code for voice selection.

        Returns:
            Audio bytes (MP3 from edge-tts, or silent WAV in mock mode).
            Returns empty bytes if synthesis fails.
        """
        if not text or not text.strip():
            logger.warning("tts_empty_text_skipped")
            return b""

        voice = self.voice_for(language)
        logger.info("tts_synthesize", voice=voice, text_preview=text[:60], language=language)

        if self._mock:
            logger.debug("tts_mock_synthesis")
            return _build_silent_wav(300)

        try:
            import edge_tts  # type: ignore
            communicate = edge_tts.Communicate(text=text, voice=voice)
            audio_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.write(chunk["data"])
            audio_bytes = audio_buf.getvalue()
            logger.info("tts_synthesize_done", bytes_produced=len(audio_bytes))
            return audio_bytes
        except ImportError:
            logger.error("tts_edge_tts_not_installed")
            return b""
        except Exception as exc:
            logger.error("tts_synthesize_error", error=str(exc))
            return b""

    def synthesize_sync(self, text: str, language: str = "en") -> bytes:
        """Synchronous wrapper for synthesize() for use in non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.synthesize(text, language))
                    return future.result(timeout=30)
            return loop.run_until_complete(self.synthesize(text, language))
        except Exception as exc:
            logger.error("tts_sync_error", error=str(exc))
            return b""
