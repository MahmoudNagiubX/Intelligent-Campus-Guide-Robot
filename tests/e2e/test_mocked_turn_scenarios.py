from __future__ import annotations

import asyncio
import time

import pytest

from app.actions.navigation_bridge import NavigationBridge
from app.audio.session_manager import SessionManager
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.storage.db import get_db
from app.tts.playback import PlaybackState
from app.utils.contracts import IntentClass, SessionState
from tests.fixtures.runtime_fixtures import (
    DummyTextGroq,
    bootstrap_and_sync,
    configure_test_settings,
    make_router_mock,
    simulate_user_turn,
)


SCENARIOS = [
    (
        "Hey Navigator, where is the Robotics Lab?",
        IntentClass.CAMPUS_QUERY,
        "Robotics Lab",
        False,
    ),
    (
        "Hey Navigator, take me to the Software Engineering Department.",
        IntentClass.NAVIGATION_REQUEST,
        "Software Engineering Department",
        True,
    ),
    (
        "Hey Navigator, how are you?",
        IntentClass.SOCIAL_CHAT,
        None,
        False,
    ),
    (
        "Hey Navigator, where is Dr Ahmed's office?",
        IntentClass.CAMPUS_QUERY,
        "Dr Ahmed",
        False,
    ),
    (
        "Hey Navigator, take me to Lab 214.",
        IntentClass.NAVIGATION_REQUEST,
        "Lab 214",
        True,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("utterance", "intent", "target_text", "expects_navigation"),
    SCENARIOS,
)
async def test_mocked_end_to_end_turn_scenarios(
    monkeypatch,
    tmp_path,
    utterance,
    intent,
    target_text,
    expects_navigation,
):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    conn = get_db()
    lab_id = conn.execute("SELECT id FROM locations WHERE code='LAB_214';").fetchone()["id"]
    se_dept_id = conn.execute("SELECT id FROM departments WHERE code='SE_DEPT';").fetchone()["id"]
    conn.executemany(
        "INSERT INTO navigation_targets (target_type, canonical_id, nav_code) VALUES (?, ?, ?);",
        [
            ("location", lab_id, "NAV_LAB_214"),
            ("department", se_dept_id, "NAV_SE_DEPT"),
        ],
    )
    conn.commit()

    router_groq = make_router_mock(intent, target_text=target_text, confidence=0.97)
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

    if expects_navigation:
        monkeypatch.setattr("app.actions.command_bus.httpx.post", fake_post)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("Scenario handled.")),
        navigation_bridge=NavigationBridge(mock=not expects_navigation),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, utterance)
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = [event.name for event in runtime.tracer.events()]
        assert "response_generated" in events

        if intent == IntentClass.SOCIAL_CHAT:
            assert "retrieval_finished" not in events
        else:
            assert "retrieval_finished" in events

        if expects_navigation:
            assert "action_emitted" in events
            assert captured_payload["action"] == "navigate"
            assert captured_payload["target_code"]
        else:
            assert "action_emitted" not in events
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_mocked_end_to_end_user_interruption_while_speaking(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.SOCIAL_CHAT, confidence=0.96)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("Long enough answer for interruption.")),
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
        await simulate_user_turn(runtime, "Hey Navigator, how are you?")
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
