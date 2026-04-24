from __future__ import annotations

import asyncio
import json
import time

import pytest

from app.actions.navigation_bridge import NavigationBridge
from app.audio.session_manager import SessionManager
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.storage.db import get_db
from app.tts.playback import PlaybackState
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import (
    bootstrap_and_sync,
    configure_test_settings,
    simulate_user_turn,
)


class ScenarioRouterGroq:
    def complete_json(self, *args, **kwargs) -> str:
        transcript = kwargs.get("user_message", "").lower()
        if "take me" in transcript:
            return json.dumps(
                {
                    "intent": "Navigation_Request",
                    "confidence": 0.97,
                    "language": "en",
                    "needs_retrieval": True,
                    "needs_navigation": True,
                    "target_entity": "Robotics Lab",
                    "target_type": "room",
                    "normalized_query": transcript,
                    "clarification_needed": False,
                    "clarification_question": "",
                }
            )
        if "where is" in transcript:
            return json.dumps(
                {
                    "intent": "Campus_Query",
                    "confidence": 0.97,
                    "language": "en",
                    "needs_retrieval": True,
                    "needs_navigation": False,
                    "target_entity": "Robotics Lab",
                    "target_type": "room",
                    "normalized_query": transcript,
                    "clarification_needed": False,
                    "clarification_question": "",
                }
            )
        return json.dumps(
            {
                "intent": "Social_Chat",
                "confidence": 0.97,
                "language": "en",
                "needs_retrieval": False,
                "needs_navigation": False,
                "target_entity": "",
                "target_type": "none",
                "normalized_query": transcript,
                "clarification_needed": False,
                "clarification_question": "",
            }
        )


class ScenarioTextGroq:
    def complete_text(self, *args, **kwargs) -> str:
        return "Scenario handled."


@pytest.mark.asyncio
async def test_mocked_end_to_end_turn_scenarios(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: ScenarioRouterGroq())

    conn = get_db()
    room_id = conn.execute(
        "SELECT id FROM rooms WHERE room_name='Robotics Lab' AND lang='en' ORDER BY id LIMIT 1"
    ).fetchone()["id"]
    conn.execute(
        """
        INSERT INTO navigation_targets (target_type, canonical_id, nav_code, updated_at)
        VALUES ('room', ?, 'NAV_C105', datetime('now'))
        ON CONFLICT(target_type, canonical_id) DO UPDATE SET
            nav_code=excluded.nav_code,
            updated_at=excluded.updated_at
        """,
        (room_id,),
    )
    conn.commit()

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
        controller=ConversationController(groq=ScenarioTextGroq()),
        navigation_bridge=NavigationBridge(mock=False),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        await asyncio.sleep(2.1)
        await simulate_user_turn(runtime, "take me to the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        await asyncio.sleep(2.1)
        await simulate_user_turn(runtime, "hello navigator")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = [event.name for event in runtime.tracer.events()]
        assert "response_generated" in events
        assert "retrieval_finished" in events
        assert "action_emitted" in events
        assert captured_payload["target_code"] == "NAV_LAB_ROBOTICS_AND_MACHINE_VISION"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_mocked_end_to_end_vad_is_muted_while_speaking(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: ScenarioRouterGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=ScenarioTextGroq()),
        session_manager=SessionManager(session_timeout_sec=3),
    )

    def slow_mock_playback():
        for _ in range(30):
            time.sleep(0.02)
            if runtime.playback_manager._stop_event.is_set():
                with runtime.playback_manager._lock:
                    runtime.playback_manager._state = PlaybackState.STOPPED
                return

    runtime.playback_manager._mock_playback = slow_mock_playback  # type: ignore[attr-defined]

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "hello navigator")
        assert await runtime.wait_for_state(SessionState.SPEAKING, timeout=2.0)

        runtime.vad.set_mock_speech(True)
        runtime.process_audio_frame(b"\x01" * 1024)
        await asyncio.sleep(0.15)
        runtime.vad.set_mock_speech(False)

        events = [event.name for event in runtime.tracer.events()]
        assert runtime.session_manager.state == SessionState.LISTENING
        assert runtime.playback_manager.state == PlaybackState.STOPPED
        assert "speaking_interrupted" in events
    finally:
        await runtime.shutdown()
