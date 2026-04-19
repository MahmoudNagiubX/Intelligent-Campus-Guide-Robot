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
    latest_session_id,
    make_router_mock,
    simulate_user_turn,
)


def test_csv_sync_into_sqlite(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    results, counts = bootstrap_and_sync()

    assert results["locations"]["upserted"] > 0
    assert results["departments"]["upserted"] > 0
    assert counts["locations"] >= 10
    assert counts["staff"] >= 1
    assert counts["navigation_targets"] > 0


@pytest.mark.asyncio
async def test_final_transcript_routes_through_campus_pipeline(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.CAMPUS_QUERY,
        target_text="Robotics Lab",
        confidence=0.97,
    )
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(
            groq=DummyTextGroq("The Robotics Lab is in Building C, floor 2.")
        ),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab")
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

        observed_frames = {event.get("frame") for event in runtime.observer.snapshot() if "frame" in event}
        assert "TranscriptEventFrame" in observed_frames
        assert "ResponsePacketFrame" in observed_frames
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_navigation_request_emits_action_payload(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.NAVIGATION_REQUEST,
        target_text="Robotics Lab",
        confidence=0.98,
    )
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
        controller=ConversationController(groq=DummyTextGroq("unused")),
        navigation_bridge=NavigationBridge(mock=False),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "take me to the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = [event.name for event in runtime.tracer.events()]
        assert router_groq.complete_json.called
        assert captured_payload["target_code"] == "NAV_C105"
        assert captured_payload["action"] == "navigate"
        assert "action_emitted" in events
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_tts_interruption_stops_playback_and_returns_to_listening(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.SOCIAL_CHAT,
        target_text=None,
        confidence=0.96,
    )
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(
            groq=DummyTextGroq("Hello there. I am Navigator and I can help you.")
        ),
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
        await simulate_user_turn(runtime, "how are you")
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
