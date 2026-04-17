from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

from app.config.settings import get_settings
from app.storage import bootstrap_schema, close_db, get_table_counts
from app.storage.sync_csv import sync_all_csvs
from app.utils.contracts import IntentClass, SessionState
from app.vad.silero_vad import _END_OF_UTTERANCE_FRAMES

REPO_ROOT = Path(__file__).resolve().parents[2]


class DummyTextGroq:
    """Simple Groq stub for response-generation tests."""

    def __init__(self, text: str) -> None:
        self._text = text

    def complete_text(self, *args, **kwargs) -> str:
        return self._text

    def complete_json(self, *args, **kwargs) -> None:
        return None


def configure_test_settings(monkeypatch, tmp_path, *, session_timeout: int = 5) -> Path:
    """Point config at a temp SQLite DB while reusing the repo CSV data."""
    close_db()
    get_settings.cache_clear()

    db_path = tmp_path / "navigator_test.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("CSV_DATA_DIR", str(REPO_ROOT / "data" / "csv"))
    monkeypatch.setenv("SESSION_TIMEOUT_SEC", str(session_timeout))
    get_settings.cache_clear()
    return db_path


def bootstrap_and_sync() -> tuple[dict[str, dict], dict[str, int]]:
    """Create schema, sync CSV files, and return sync results plus counts."""
    bootstrap_schema()
    results = sync_all_csvs()
    counts = get_table_counts()
    return results, counts


def make_router_mock(
    intent: IntentClass,
    *,
    target_text: str | None = None,
    language: str = "en",
    confidence: float = 0.95,
    needs_clarification: bool = False,
    clarification_question: str | None = None,
) -> MagicMock:
    """Build a mocked router Groq client that returns a valid JSON payload."""
    payload = {
        "intent": intent.value,
        "language": language,
        "target_text": target_text,
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }
    groq = MagicMock()
    groq.complete_json.return_value = json.dumps(payload)
    return groq


async def simulate_user_turn(runtime, transcript: str, *, language: str = "en") -> None:
    """
    Simulate one spoken user turn through wake word, VAD boundaries, and STT.

    The transcript itself is injected through the mock Deepgram client after the
    VAD mock has transitioned the session into PROCESSING.
    """
    runtime.trigger_wake_word()
    assert await runtime.wait_for_state(SessionState.LISTENING, timeout=1.0)

    runtime.vad.set_mock_speech(True)
    runtime.process_audio_frame(b"\x01" * 1024)
    await asyncio.sleep(0.05)

    runtime.vad.set_mock_speech(False)
    for _ in range(_END_OF_UTTERANCE_FRAMES):
        runtime.process_audio_frame(b"\x00" * 1024)

    await asyncio.sleep(0.05)
    runtime.inject_mock_transcript(transcript, language=language)
    await asyncio.sleep(0.2)


def latest_session_id(runtime) -> str | None:
    """Return the newest non-empty session ID seen by the runtime tracer."""
    for event in reversed(runtime.tracer.events()):
        if event.session_id:
            return event.session_id
    return None
