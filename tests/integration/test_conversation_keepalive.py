"""
Integration test: multi-turn conversation keepalive.

Verifies that after TTS playback completes the session stays in LISTENING
(not IDLE), and that a follow-up question can be asked without a new wake word.
The robot returns to IDLE only after the inactivity timer expires.
"""

from __future__ import annotations

import asyncio

import pytest

from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import (
    bootstrap_and_sync,
    configure_test_settings,
    simulate_user_turn,
)


class _SimpleGroq:
    def complete_text(self, *args, **kwargs) -> str:
        return "The robotics lab is in building C, room 105."

    def complete_json(self, *args, **kwargs) -> str:
        import json

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


@pytest.mark.asyncio
async def test_session_stays_listening_after_playback(monkeypatch, tmp_path):
    """After one answer, session must remain in LISTENING — not transition to IDLE."""
    configure_test_settings(monkeypatch, tmp_path, session_timeout=5)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: _SimpleGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=_SimpleGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        # Give the mock playback (100 ms) time to complete
        await asyncio.sleep(0.5)

        assert runtime.session_manager.state == SessionState.LISTENING, (
            "Session should stay LISTENING after playback, not return to IDLE"
        )
        assert runtime.session_manager.session_id is not None
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_followup_question_without_wake_word(monkeypatch, tmp_path):
    """Follow-up question must work without a new wake word in the same session."""
    configure_test_settings(monkeypatch, tmp_path, session_timeout=5)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: _SimpleGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=_SimpleGroq()),
    )

    await runtime.start()
    try:
        # First turn
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        await asyncio.sleep(0.5)

        first_session_id = runtime.session_manager.session_id
        assert runtime.session_manager.state == SessionState.LISTENING
        assert first_session_id is not None

        # Second turn — inject speech directly (no trigger_wake_word)
        runtime.vad.set_mock_speech(True)
        runtime.process_audio_frame(b"\x01" * 1024)
        await asyncio.sleep(0.05)

        runtime.vad.set_mock_speech(False)
        for _ in range(runtime.vad._end_of_utterance_frames):
            runtime.process_audio_frame(b"\x00" * 1024)

        await asyncio.sleep(0.05)
        runtime.inject_mock_transcript("what floor is it on", language="en")
        await asyncio.sleep(0.5)

        # Session should still be alive (LISTENING again after second answer)
        assert runtime.session_manager.state == SessionState.LISTENING
        assert runtime.session_manager.session_id == first_session_id, (
            "Follow-up should use the same session, not start a new one"
        )
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_session_ends_after_inactivity_timeout(monkeypatch, tmp_path):
    """After session_timeout_sec of no speech, robot must return to IDLE."""
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: _SimpleGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=_SimpleGroq()),
    )

    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        # Wait for mock playback to complete
        await asyncio.sleep(0.5)
        assert runtime.session_manager.state == SessionState.LISTENING

        # Wait for the 1-second inactivity timeout to fire
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0), (
            "Session must return to IDLE after inactivity timeout"
        )
    finally:
        await runtime.shutdown()
