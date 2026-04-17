from __future__ import annotations

import asyncio

import pytest

from app.audio.session_manager import SessionManager
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import IntentClass, SessionState
from tests.fixtures.runtime_fixtures import (
    DummyTextGroq,
    bootstrap_and_sync,
    configure_test_settings,
    latest_session_id,
    make_router_mock,
    simulate_user_turn,
)


class FailingTextGroq:
    def complete_text(self, *args, **kwargs):
        raise RuntimeError("timeout")

    def complete_json(self, *args, **kwargs):
        return None


class SilentTTS:
    async def synthesize(self, text: str, language: str = "en") -> bytes:
        return b""


@pytest.mark.asyncio
async def test_groq_timeout_falls_back_to_grounded_facts(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.CAMPUS_QUERY,
        target_text="Robotics Lab",
        confidence=0.95,
    )
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

        response_events = [e for e in runtime.tracer.events() if e.name == "response_generated"]
        assert response_events
        assert "Robotics Lab" in response_events[-1].data["text"]
        assert "Building C" in response_events[-1].data["text"]
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_tts_failure_records_error_and_resets_session(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(IntentClass.SOCIAL_CHAT, confidence=0.96)
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("Hello from Navigator.")),
        tts_client=SilentTTS(),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "how are you")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        events = runtime.tracer.events()
        assert any(event.name == "error_occurred" and event.data["source"] == "tts" for event in events)
        assert any(event.name == "session_ended" and event.data["reason"] == "empty_audio" for event in events)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_db_failure_returns_safe_response(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.CAMPUS_QUERY,
        target_text="Robotics Lab",
        confidence=0.95,
    )
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)
    monkeypatch.setattr("app.pipeline.controller.retrieve", lambda query: (_ for _ in ()).throw(RuntimeError("db missing")))

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("unused")),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        response_events = [e for e in runtime.tracer.events() if e.name == "response_generated"]
        assert response_events
        assert "Something went wrong" in response_events[-1].data["text"]
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_retrieval_not_found_returns_safe_bounded_answer(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path)
    bootstrap_and_sync()

    router_groq = make_router_mock(
        IntentClass.CAMPUS_QUERY,
        target_text="Quantum Mechanics Lab",
        confidence=0.95,
    )
    monkeypatch.setattr("app.routing.router._get_groq", lambda: router_groq)

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("unused")),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the quantum mechanics lab")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)

        response_events = [e for e in runtime.tracer.events() if e.name == "response_generated"]
        assert response_events
        assert "couldn't find" in response_events[-1].data["text"].lower()
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_wake_word_false_trigger_times_out_cleanly(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=DummyTextGroq("unused")),
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
            if any(
                event.name == "session_ended" and event.data["reason"] == "timeout"
                for event in events
            ):
                break

            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("Expected timeout session_ended trace event before assertion deadline.")

            await asyncio.sleep(0.02)
    finally:
        await runtime.shutdown()
