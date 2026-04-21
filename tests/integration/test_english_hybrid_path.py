from __future__ import annotations

import json

import pytest

from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import bootstrap_and_sync, configure_test_settings, simulate_user_turn


class EnglishHybridGroq:
    def complete_json(self, *args, **kwargs) -> str:
        text = kwargs.get("user_message", "").lower()
        if "what is ecu known for" in text:
            intent = "Unknown"
            target = ""
        elif "faculties" in text:
            intent = "Campus_Query"
            target = "faculties"
        else:
            intent = "Campus_Query"
            target = "Robotics Lab"
        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.9,
                "language": "en",
                "needs_retrieval": intent != "Unknown",
                "needs_navigation": False,
                "target_entity": target,
                "target_type": "none",
                "normalized_query": text,
                "clarification_needed": False,
                "clarification_question": "",
            }
        )

    def complete_text(self, *args, **kwargs) -> str:
        prompt = kwargs.get("system_prompt", "")
        message = kwargs.get("user_message", "")
        if "Faculties:" in prompt or "faculties" in message.lower():
            return "ECU has faculties including Engineering, Computers and Information Systems, Pharmacy, and Law."
        if "not in your campus database" in prompt:
            return "ECU is known for offering programs across several faculties, including Engineering and Computer Science."
        return "The Robotics Lab is in Building A, first floor, room C105."


@pytest.mark.asyncio
async def test_english_exact_db_match(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: EnglishHybridGroq())
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishHybridGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any("room" in text.lower() or "building" in text.lower() for text in responses)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_english_semantic_db_match(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: EnglishHybridGroq())
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishHybridGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "tell me about the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any("not found" not in text.lower() and "room" in text.lower() for text in responses)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_english_ecu_cache_fallback(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: EnglishHybridGroq())
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishHybridGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "what faculties does ECU have?", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any("facult" in text.lower() for text in responses)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_english_general_campus_fallback(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: EnglishHybridGroq())
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=EnglishHybridGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "what is ECU known for?", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any("not found" not in text.lower() and "ECU" in text for text in responses)
    finally:
        await runtime.shutdown()
