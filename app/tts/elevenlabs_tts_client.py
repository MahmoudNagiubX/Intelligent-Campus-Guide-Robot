"""
Arabic ElevenLabs TTS client.

Only used when ELEVENLABS_TTS_ARABIC_ENABLED=true and a voice ID is present.
Callers should fall back to edge-tts whenever this client returns empty bytes.
"""

from __future__ import annotations

from app.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TIMEOUT_SEC = 8.0


class ElevenLabsTTSClient:
    """ElevenLabs TTS wrapper for Arabic output."""

    def __init__(self, mock: bool = False) -> None:
        cfg = get_settings()
        self._api_key = cfg.elevenlabs_api_key
        self._voice_id = cfg.elevenlabs_tts_voice_ar
        self._mock = mock
        self._enabled = bool(cfg.elevenlabs_tts_arabic_enabled and self._api_key and self._voice_id)

        masked_voice = f"{self._voice_id[:8]}..." if self._voice_id else "none"
        logger.info("elevenlabs_tts_init", enabled=self._enabled, voice_id=masked_voice, mock=self._mock)

    @property
    def is_enabled(self) -> bool:
        return self._enabled and not self._mock

    async def synthesize(self, text: str) -> bytes:
        """Return MP3 bytes, or empty bytes when ElevenLabs cannot be used."""
        if self._mock or not self.is_enabled or not text or not text.strip():
            return b""

        try:
            import httpx
        except ImportError:
            logger.error("elevenlabs_tts_httpx_missing")
            return b""

        url = _ELEVENLABS_TTS_URL.format(voice_id=self._voice_id)
        headers = {
            "xi" + "-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.65,
                "similarity_boost": 0.80,
                "style": 0.35,
                "use_speaker_boost": True,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SEC) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                logger.debug("elevenlabs_tts_ok", chars=len(text), bytes=len(response.content))
                return response.content
        except httpx.TimeoutException:
            logger.warning("elevenlabs_tts_timeout", chars=len(text))
            return b""
        except httpx.HTTPStatusError as exc:
            logger.error(
                "elevenlabs_tts_http_error",
                status=exc.response.status_code,
                detail=exc.response.text[:120],
            )
            return b""
        except Exception as exc:
            logger.error("elevenlabs_tts_unexpected_error", error=str(exc))
            return b""
