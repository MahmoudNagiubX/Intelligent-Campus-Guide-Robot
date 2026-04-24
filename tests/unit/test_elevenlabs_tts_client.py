import types

import pytest

from app.config.settings import get_settings
from app.tts.elevenlabs_tts_client import ElevenLabsTTSClient


def test_elevenlabs_tts_disabled_without_voice(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_TTS_ARABIC_ENABLED", "true")
    monkeypatch.setenv("ELEVENLABS_TTS_VOICE_AR", "")
    get_settings.cache_clear()
    try:
        client = ElevenLabsTTSClient(mock=False)
        assert client.is_enabled is False
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_elevenlabs_tts_returns_empty_on_http_failure(monkeypatch) -> None:
    monkeypatch.setenv("ELEVENLABS_TTS_ARABIC_ENABLED", "true")
    monkeypatch.setenv("ELEVENLABS_TTS_VOICE_AR", "voice123")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "x" * 24)
    get_settings.cache_clear()

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise RuntimeError("network down")

    fake_httpx = types.SimpleNamespace(
        AsyncClient=FakeClient,
        TimeoutException=TimeoutError,
        HTTPStatusError=type("HTTPStatusError", (Exception,), {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)
    try:
        client = ElevenLabsTTSClient(mock=False)
        assert client.is_enabled is True
        assert await client.synthesize("أهلاً") == b""
    finally:
        get_settings.cache_clear()
        __import__("sys").modules.pop("httpx", None)
