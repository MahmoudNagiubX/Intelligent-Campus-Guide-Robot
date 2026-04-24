"""
Navigator - edge-tts TTS Client
Phase 7, Step 7.1

Wraps the edge-tts library for async text-to-speech synthesis.
Selects the correct neural voice based on the response language.

Supported voices (locked MVP):
- English: en-US-ChristopherNeural
- Arabic:  ar-EG-ShakirNeural

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
import re
import wave

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06FF]")
_TTS_MAX_ATTEMPTS = 3
_TTS_RETRY_DELAYS = [0.0, 0.4, 0.8]
_fallback_audio_cache: bytes = b""

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
        self._voice_en = cfg.edge_tts_voice_en    # "en-US-ChristopherNeural"
        self._voice_ar = cfg.edge_tts_voice_ar    # "ar-EG-SalmaNeural"
        self._rate = cfg.edge_tts_rate
        self._rate_ar = cfg.edge_tts_rate_ar
        self._mock = mock

        logger.info(
            "tts_client_init",
            voice_en=self._voice_en,
            voice_ar=self._voice_ar,
            rate=self._rate,
            rate_ar=self._rate_ar,
            mock=self._mock,
        )

    def voice_for(self, language: str, text: str = "") -> str:
        """Return the locked neural voice name for the given language code."""
        if get_settings().english_only_mode:
            return self._voice_en
        if language.startswith("ar"):
            return self._voice_ar
        if text and _ARABIC_SCRIPT.search(text):
            logger.warning(
                "tts_arabic_script_detected_with_wrong_language",
                language=language,
                text_preview=text[:40],
            )
            return self._voice_ar
        return self._voice_en

    def rate_for(self, language: str, text: str = "") -> str:
        """Return the speech rate for the given language code."""
        if get_settings().english_only_mode:
            return self._rate
        if language.startswith("ar") or (text and _ARABIC_SCRIPT.search(text)):
            return self._rate_ar
        return self._rate

    async def prewarm_fallback(self) -> None:
        """
        Pre-synthesize the fallback phrase once and cache it in memory.

        This gives the runtime something audible to play even when a later
        network TTS request fails completely.
        """
        global _fallback_audio_cache
        if _fallback_audio_cache:
            return

        if self._mock:
            _fallback_audio_cache = _build_silent_wav(800)
            logger.info("tts_fallback_cached", bytes=len(_fallback_audio_cache), mock=True)
            return

        phrase = get_settings().tts_fallback_phrase
        try:
            audio = await self._synthesize_once(phrase, "en", self._voice_en, self._rate)
            if audio:
                _fallback_audio_cache = audio
                logger.info("tts_fallback_cached", bytes=len(audio))
                return
            logger.warning("tts_fallback_prewarm_failed_using_silence")
        except Exception as exc:
            logger.warning("tts_fallback_prewarm_error", error=str(exc))

        _fallback_audio_cache = _build_silent_wav(800)

    async def _synthesize_once(self, text: str, language: str, voice: str, rate: str) -> bytes:
        """Run one edge-tts synthesis attempt and return empty bytes on service failure."""
        del language
        try:
            import edge_tts  # type: ignore

            communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
            audio_buf = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.write(chunk["data"])
            return audio_buf.getvalue()
        except ImportError:
            logger.error("tts_edge_tts_not_installed")
            raise
        except Exception as exc:
            logger.warning("tts_single_attempt_failed", error=str(exc), voice=voice)
            return b""

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        """
        Synthesize text to speech with automatic retry on empty response.

        Microsoft edge-tts drops the stream after several rapid calls.
        Three attempts with increasing backoff recovers from most failures.
        """
        if not text or not text.strip():
            logger.warning("tts_empty_text_skipped")
            return b""

        if self._mock:
            return _build_silent_wav(300)

        voice = self.voice_for(language, text=text)
        rate = self.rate_for(language, text=text)
        has_arabic_script = bool(_ARABIC_SCRIPT.search(text))
        if has_arabic_script and not language.startswith("ar"):
            logger.warning(
                "tts_language_text_mismatch",
                text_script="arabic",
                language_param=language,
                text_preview=text[:60],
                action="switching_voice_to_arabic",
            )
            voice = self._voice_ar
            rate = self._rate_ar
        elif not has_arabic_script and language.startswith("ar") and len(text.strip()) > 5:
            logger.warning(
                "tts_language_text_mismatch",
                text_script="latin",
                language_param=language,
                text_preview=text[:60],
                action="keeping_arabic_voice_for_mixed_content",
            )
        logger.info(
            "tts_synthesize",
            voice=voice,
            rate=rate,
            text_preview=text[:60],
            language=language,
        )

        last_exc: Exception | None = None

        for attempt in range(_TTS_MAX_ATTEMPTS):
            delay = _TTS_RETRY_DELAYS[attempt]
            if delay > 0:
                await asyncio.sleep(delay)
                logger.warning("tts_retry", attempt=attempt, voice=voice, text_preview=text[:40])
            try:
                audio_bytes = await self._synthesize_once(text, language, voice, rate)

                if audio_bytes:
                    if attempt > 0:
                        logger.info("tts_retry_succeeded", attempt=attempt, bytes_produced=len(audio_bytes))
                    else:
                        logger.info("tts_synthesize_done", bytes_produced=len(audio_bytes))
                    return audio_bytes

                last_exc = ValueError(f"edge-tts returned empty stream on attempt {attempt}")
                logger.warning("tts_empty_stream", attempt=attempt, voice=voice)

            except ImportError:
                return self._fallback_audio()
            except Exception as exc:
                last_exc = exc
                logger.warning("tts_attempt_failed", attempt=attempt, error=str(exc), voice=voice)

        logger.error(
            "tts_all_attempts_failed",
            attempts=_TTS_MAX_ATTEMPTS,
            error=str(last_exc),
            voice=voice,
            text_preview=text[:60],
        )
        return self._fallback_audio()

    @staticmethod
    def _fallback_audio() -> bytes:
        """Return cached fallback audio, or a tiny valid WAV as the last resort."""
        if _fallback_audio_cache:
            logger.warning("tts_using_fallback_audio")
            return _fallback_audio_cache
        logger.error("tts_no_fallback_available_returning_silence")
        return _build_silent_wav(500)

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
