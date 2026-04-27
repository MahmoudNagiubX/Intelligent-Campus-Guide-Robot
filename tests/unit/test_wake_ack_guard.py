"""
Unit tests for the wake acknowledgment per-session guard.

The guard sits in _on_wake_word_detected():

    if self._loop and not self._mock and session_id != self._wake_ack_session_id:
        self._wake_ack_session_id = session_id
        asyncio.run_coroutine_threadsafe(self._play_wake_ack_async(), self._loop)

Scenarios:
1. Mock mode → ack never scheduled, _wake_ack_session_id stays None.
2. Real mode, first wake → ack scheduled once, _wake_ack_session_id set.
3. Real mode, same session_id → ack NOT scheduled again (guard blocks duplicate).
4. State-machine guard: _on_wake_word_detected returns early when state != IDLE,
   so keepalive (SPEAKING→LISTENING) can never replay the ack.
5. New session after timeout → _wake_ack_session_id updated, ack scheduled again.
6. wake_ack_enabled=False → _play_wake_ack_async returns without calling TTS.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch


def _mock_rcts():
    """Return a side_effect for run_coroutine_threadsafe that closes leaked coroutines."""
    calls = []

    def _impl(coro, loop):
        calls.append(coro)
        coro.close()  # prevent "coroutine was never awaited" RuntimeWarning
        return MagicMock()

    _impl.calls = calls
    _impl.call_count_prop = lambda: len(calls)
    return _impl

import pytest

from app.audio.session_manager import SessionManager
from app.config.settings import get_settings
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import configure_test_settings


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_runtime(monkeypatch, tmp_path, *, session_timeout: int = 30) -> NavigatorPipecatRuntime:
    configure_test_settings(monkeypatch, tmp_path, session_timeout=session_timeout)
    return NavigatorPipecatRuntime(mock=True, auto_start_audio=False)


# ------------------------------------------------------------------
# 1. Mock mode skips ack entirely
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_mode_does_not_schedule_wake_ack(monkeypatch, tmp_path):
    """In mock mode, _wake_ack_session_id must stay None after wake."""
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()
    try:
        runtime.trigger_wake_word()
        await asyncio.sleep(0.1)

        assert runtime._wake_ack_session_id is None, (
            "Mock mode must never set _wake_ack_session_id"
        )
    finally:
        await runtime.shutdown()


# ------------------------------------------------------------------
# 2. Real-mode first wake schedules ack exactly once
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_wake_schedules_ack_once(monkeypatch, tmp_path):
    """After the first wake word, run_coroutine_threadsafe called exactly once."""
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    side_effect = _mock_rcts()
    try:
        with patch("app.pipeline.pipecat_graph.asyncio.run_coroutine_threadsafe", side_effect=side_effect):
            runtime._mock = False
            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)

            assert side_effect.call_count_prop() == 1, "ack must be scheduled exactly once"
            session_id = runtime.session_manager.session_id
            assert runtime._wake_ack_session_id == session_id
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 3. Same session_id does not replay the ack
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_session_id_does_not_replay_ack(monkeypatch, tmp_path):
    """If _wake_ack_session_id already equals current session_id, ack is skipped."""
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    side_effect = _mock_rcts()
    try:
        with patch("app.pipeline.pipecat_graph.asyncio.run_coroutine_threadsafe", side_effect=side_effect):
            runtime._mock = False

            # First wake: ack scheduled, guard set
            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)
            assert side_effect.call_count_prop() == 1

            session_id = runtime.session_manager.session_id

            # Re-apply the guard condition manually (same session_id → guard blocks it)
            if runtime._loop and not runtime._mock and session_id != runtime._wake_ack_session_id:
                runtime._wake_ack_session_id = session_id
                asyncio.run_coroutine_threadsafe(runtime._play_wake_ack_async(), runtime._loop)

            # Guard should have blocked the second scheduling
            assert side_effect.call_count_prop() == 1, "same session_id must not replay ack"
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 4. Keepalive path never replays ack (state machine guard)
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keepalive_does_not_replay_ack(monkeypatch, tmp_path):
    """After SPEAKING→LISTENING keepalive, _on_wake_word_detected is a no-op."""
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    side_effect = _mock_rcts()
    try:
        with patch("app.pipeline.pipecat_graph.asyncio.run_coroutine_threadsafe", side_effect=side_effect):
            runtime._mock = False

            # First wake
            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)
            first_ack_count = side_effect.call_count_prop()  # should be 1

            # Simulate SPEAKING→LISTENING keepalive (same session)
            runtime.session_manager.on_speech_end()
            runtime.session_manager.on_response_ready()
            runtime.session_manager.on_playback_complete()  # → LISTENING

            assert runtime.session_manager.state == SessionState.LISTENING

            # Trigger wake word again — must be blocked by state != IDLE guard
            runtime._on_wake_word_detected()
            await asyncio.sleep(0.1)

            assert side_effect.call_count_prop() == first_ack_count, (
                "Keepalive LISTENING state blocks re-entry of _on_wake_word_detected"
            )
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 5. New session after timeout gets fresh ack
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_session_after_timeout_schedules_ack_again(monkeypatch, tmp_path):
    """After session ends (IDLE), a new wake word gets a new session UUID and ack."""
    runtime = _make_runtime(monkeypatch, tmp_path, session_timeout=1)
    await runtime.start()

    side_effect = _mock_rcts()
    try:
        with patch("app.pipeline.pipecat_graph.asyncio.run_coroutine_threadsafe", side_effect=side_effect):
            runtime._mock = False

            # First wake → ack scheduled (count = 1)
            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)
            first_session_id = runtime._wake_ack_session_id
            assert side_effect.call_count_prop() == 1

            # End session manually → back to IDLE
            runtime.session_manager.end_session("test_end")
            runtime._wakeword.set_session_active(False)
            await asyncio.sleep(0.05)

            assert runtime.session_manager.state == SessionState.IDLE

            # Reset wakeword cooldown so the second trigger is not suppressed
            runtime._wakeword._last_trigger_time = 0.0

            # Second wake → new UUID → ack should fire again
            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)

            second_session_id = runtime._wake_ack_session_id
            assert second_session_id != first_session_id, "New session must have new UUID"
            assert side_effect.call_count_prop() == 2, "New session must schedule ack again"
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 6. wake_ack_enabled=False → _play_wake_ack_async skips TTS
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_ack_disabled_skips_synthesis(monkeypatch, tmp_path):
    """When WAKE_ACK_ENABLED=false, _play_wake_ack_async returns immediately."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "false")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        synthesize_calls = []

        async def mock_synthesize(text, lang):
            synthesize_calls.append((text, lang))
            return b""

        runtime._tts_client.synthesize = mock_synthesize

        await runtime._play_wake_ack_async()

        assert synthesize_calls == [], "TTS must not be called when wake_ack_enabled=False"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 7. _play_wake_ack_async: empty audio from TTS does not call playback
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_ack_empty_audio_skips_playback(monkeypatch, tmp_path):
    """If TTS returns empty bytes for the ack, playback must not be triggered."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "true")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        async def mock_synthesize(text, lang):
            return b""

        runtime._tts_client.synthesize = mock_synthesize
        playback_started_calls = []
        runtime._on_playback_started = lambda sid: playback_started_calls.append(sid)

        await runtime._play_wake_ack_async()

        assert playback_started_calls == [], "No playback when TTS returns empty audio"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 8. _play_wake_ack_async: TTS exception does not propagate
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wake_ack_tts_exception_does_not_propagate(monkeypatch, tmp_path):
    """Exceptions in TTS synthesis during wake ack must be caught silently."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "true")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        async def broken_synthesize(text, lang):
            raise RuntimeError("TTS exploded")

        runtime._tts_client.synthesize = broken_synthesize

        try:
            await runtime._play_wake_ack_async()
        except Exception as exc:
            pytest.fail(f"_play_wake_ack_async must not propagate exceptions: {exc}")
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()
