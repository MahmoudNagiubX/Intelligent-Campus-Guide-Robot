from __future__ import annotations

import asyncio
import os
import time

import pytest

from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from tests.fixtures.runtime_fixtures import bootstrap_and_sync, configure_test_settings, latest_session_id


@pytest.mark.asyncio
@pytest.mark.hardware
async def test_real_hardware_pipeline_roundtrip(monkeypatch, tmp_path):
    """
    Manual hardware-gated voice roundtrip.

    This test is skipped by default and is intended to be run on the Raspberry
    Pi with a real microphone, speaker, Deepgram key, and Groq key.
    """
    if os.getenv("NAVIGATOR_RUN_HARDWARE_E2E") != "1":
        pytest.skip("Set NAVIGATOR_RUN_HARDWARE_E2E=1 to run the real hardware pipeline test.")

    if not os.getenv("DEEPGRAM_API_KEY") or not os.getenv("GROQ_API_KEY"):
        pytest.skip("Deepgram and Groq API keys are required for the hardware pipeline test.")

    configure_test_settings(monkeypatch, tmp_path, session_timeout=20)
    bootstrap_and_sync()

    runtime = NavigatorPipecatRuntime(mock=False, auto_start_audio=True)

    await runtime.start()
    try:
        print("Say: 'Hey Jarvis, where is the Robotics Lab?' within 60 seconds.")
        deadline = time.monotonic() + 60.0

        while time.monotonic() < deadline:
            events = [event.name for event in runtime.tracer.events()]
            if {"transcript_final_received", "intent_decided", "response_generated", "speaking_started", "session_ended"} <= set(events):
                break
            await asyncio.sleep(0.25)
        else:
            pytest.fail("No full hardware turn completed within 60 seconds.")

        session_id = latest_session_id(runtime)
        metrics = runtime.tracer.metrics_for(session_id)
        assert session_id is not None
        assert "full_turn_latency_sec" in metrics
    finally:
        await runtime.shutdown()
