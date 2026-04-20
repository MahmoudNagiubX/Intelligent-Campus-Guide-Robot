from __future__ import annotations

import asyncio
import json

import pytest

from app.actions.navigation_bridge import NavigationBridge
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.storage.db import get_db
from app.tts.edge_tts_client import EdgeTTSClient, _build_silent_wav
from app.utils.contracts import IntentClass, SessionState
from tests.fixtures.runtime_fixtures import (
    bootstrap_and_sync,
    configure_test_settings,
    latest_session_id,
    make_router_mock,
    simulate_user_turn,
)


class EnglishCampusGroq:
    def complete_text(self, *args, **kwargs) -> str:
        return "The robotics lab is in building C."


class BilingualCampusGroq:
    def complete_text(self, *args, **kwargs) -> str:
        system_prompt = kwargs.get("system_prompt", "")
        return "المعمل في المبنى C." if any(ord(char) > 127 for char in system_prompt) else "The lab is in building C."


class BilingualRouterGroq:
    def complete_json(self, *args, **kwargs) -> str:
        transcript = kwargs.get("user_message", "")
        if any(ord(char) > 127 for char in transcript):
            return json.dumps(
                {
                    "intent": "Campus_Query",
                    "confidence": 0.97,
                    "language": "ar",
                    "needs_retrieval": True,
                    "needs_navigation": False,
                    "target_entity": "معمل الروبوتات",
                    "target_type": "room",
                    "normalized_query": "فين معمل الروبوتات",
                    "clarification_needed": False,
                    "clarification_question": "",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "intent": "Campus_Query",
                "confidence": 0.97,
                "language": "en",
                "needs_retrieval": True,
                "needs_navigation": False,
                "target_entity": "Robotics Lab",
                "target_type": "room",
                "normalized_query": "where is the robotics lab",
                "clarification_needed": False,
                "clarification_question": "",
            }
        )


class RecordingTTS:
    def __init__(self) -> None:
        self._voices = EdgeTTSClient(mock=True)
        self.languages: list[str] = []
        self.voice_names: list[str] = []

    async def synthesize(self, text: str, language: str = "en") -> bytes:
        self.languages.append(language)
        self.voice_names.append(self._voices.voice_for(language))
        return _build_silent_wav(80)


def test_csv_sync_into_sqlite(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    results, counts = bootstrap_and_sync()

    assert results["rooms_en"]["upserted"] > 0
    assert results["rooms_ar"]["upserted"] > 0
    assert results["departments_en"]["upserted"] > 0
    assert counts["rooms"] > 0
    assert counts["departments"] > 0
    assert counts["staff"] > 0
    assert counts["navigation_targets"] == 0


@pytest.mark.asyncio
async def test_final_transcript_routes_through_campus_pipeline(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.CAMPUS_QUERY, target_text="Robotics Lab", confidence=0.97)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishCampusGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = [event.name for event in runtime.tracer.events()]
        session_id = latest_session_id(runtime)
        metrics = runtime.tracer.metrics_for(session_id)

        assert router_groq.complete_json.called
        assert "deepgram_connected" in events
        assert "transcript_final_received" in events
        assert "intent_decided" in events
        assert "retrieval_finished" in events
        assert "response_generated" in events
        assert "speaking_started" in events
        assert "session_ended" in events
        assert "full_turn_latency_sec" in metrics
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_navigation_request_emits_action_payload(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    conn = get_db()
    room_id = conn.execute(
        "SELECT id FROM rooms WHERE room_name='Robotics Lab' AND lang='en' ORDER BY id LIMIT 1"
    ).fetchone()["id"]
    conn.execute(
        """
        INSERT INTO navigation_targets (target_type, canonical_id, nav_code, updated_at)
        VALUES ('room', ?, 'NAV_C105', datetime('now'))
        """,
        (room_id,),
    )
    conn.commit()

    router_groq = make_router_mock(IntentClass.NAVIGATION_REQUEST, target_text="Robotics Lab", confidence=0.98)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    captured_payload = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "accepted"}

    def fake_post(url, json=None, timeout=None):
        captured_payload.update(json or {})
        return FakeResponse()

    monkeypatch.setattr("app.actions.command_bus.httpx.post", fake_post)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishCampusGroq()),
        navigation_bridge=NavigationBridge(mock=False),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "take me to the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = [event.name for event in runtime.tracer.events()]
        assert captured_payload["action"] == "navigate"
        assert captured_payload["target_code"] == "NAV_C105"
        assert "action_emitted" in events
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_integration_smoke_covers_english_and_arabic_tts_paths(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: BilingualRouterGroq())

    tts = RecordingTTS()
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=BilingualCampusGroq()),
        tts_client=tts,
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        await asyncio.sleep(2.1)
        await simulate_user_turn(runtime, "فين معمل الروبوتات", language="ar-EG")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        error_events = [event for event in runtime.tracer.events() if event.name == "error_occurred"]

        assert tts.languages[:2] == ["en", "ar-EG"]
        assert "Jenny" in tts.voice_names[0] or "en-US" in tts.voice_names[0]
        assert "ar-EG" in tts.voice_names[1]
        assert error_events == []
    finally:
        await runtime.shutdown()
