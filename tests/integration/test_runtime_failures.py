from __future__ import annotations

import asyncio

import pytest

from app.audio.session_manager import SessionManager
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import IntentClass, RetrievalResult, RetrievalStatus, SessionState
from tests.fixtures.runtime_fixtures import (
    bootstrap_and_sync,
    configure_test_settings,
    latest_session_id,
    make_router_mock,
    simulate_user_turn,
)


class FailingTextGroq:
    def complete_text(self, *args, **kwargs):
        raise RuntimeError("timeout")


class SilentTTS:
    async def synthesize(self, text: str, language: str = "en") -> bytes:
        return b""


@pytest.mark.asyncio
async def test_groq_timeout_falls_back_to_grounded_facts(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.CAMPUS_QUERY, target_text="Robotics Lab", confidence=0.95)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=FailingTextGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        response_events = [event for event in runtime.tracer.events() if event.name == "response_generated"]
        assert response_events
        assert "Robotics Lab" in response_events[-1].data["text"]
        assert "C105" in response_events[-1].data["text"]
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_tts_failure_records_warning_and_keeps_session_alive(monkeypatch, tmp_path):
    # Empty TTS audio no longer ends the session — the robot stays in LISTENING
    # so the user can try again. The session eventually times out normally.
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.SOCIAL_CHAT, confidence=0.96)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=FailingTextGroq()),
        tts_client=SilentTTS(),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "hello there")
        # After empty audio, session stays LISTENING (keepalive)
        await asyncio.sleep(0.3)
        assert runtime.session_manager.state == SessionState.LISTENING

        # Both tts_empty_audio events are still recorded
        events = runtime.tracer.events()
        assert any(event.name == "tts_empty_audio" and event.data["source"] == "tts" for event in events)
        assert any(event.name == "tts_empty_audio" and event.data["source"] == "playback" for event in events)

        # Session ends via inactivity timeout (not "empty_audio" reason)
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        timeout_events = [e for e in runtime.tracer.events() if e.name == "session_ended"]
        assert any(e.data.get("reason") == "timeout" for e in timeout_events)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_db_failure_returns_safe_response(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.CAMPUS_QUERY, target_text="Robotics Lab", confidence=0.95)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)
    monkeypatch.setattr(
        "app.retrieval.hybrid_retriever.search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db missing")),
    )

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=FailingTextGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        response_events = [event for event in runtime.tracer.events() if event.name == "response_generated"]
        assert response_events
        assert "Something went wrong" in response_events[-1].data["text"]
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_retrieval_not_found_returns_safe_bounded_answer(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.CAMPUS_QUERY, target_text="Quantum Mechanics Lab", confidence=0.95)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=FailingTextGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the quantum mechanics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        response_events = [event for event in runtime.tracer.events() if event.name == "response_generated"]
        assert response_events
        text = response_events[-1].data["text"].lower()
        assert "could not find" not in text
        assert "ecu.edu.eg" in text or "student affairs" in text
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_wake_word_false_trigger_times_out_cleanly(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=FailingTextGroq()),
        session_manager=SessionManager(session_timeout_sec=1),
    )

    await runtime.start()
    try:
        runtime.trigger_wake_word()
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=2.5)

        session_id = latest_session_id(runtime)
        assert session_id is not None

        deadline = asyncio.get_running_loop().time() + 0.5
        while True:
            events = runtime.tracer.events()
            if any(event.name == "session_ended" and event.data["reason"] == "timeout" for event in events):
                break
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("Expected timeout session_ended trace event before assertion deadline.")
            await asyncio.sleep(0.02)
    finally:
        await runtime.shutdown()
