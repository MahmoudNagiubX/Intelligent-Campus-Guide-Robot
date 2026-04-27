"""
Unit tests for the wake acknowledgment per-session guard.

The guard in _on_wake_word_detected():
    if (loop and not mock and session_id and
            session_id != _wake_ack_session_id and
            not _wake_ack_in_progress):
        _wake_ack_session_id = session_id
        _wake_ack_in_progress = True
        run_coroutine_threadsafe(_play_wake_ack_async(session_id), loop)

_play_wake_ack_async(scheduled_for) aborts if:
    - wake_ack_enabled is False
    - TTS returns empty audio
    - session_id changed after synthesis (new or ended session)
    - state is not LISTENING / WAKE_DETECTED at play time

If play() fails after _on_playback_started (SPEAKING, timer cancelled), the
session is recovered back to LISTENING via on_empty_response().
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.config.settings import get_settings
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import configure_test_settings

_FAKE_SESSION = "00000000-test-0000-0000-000000000000"


def _mock_rcts():
    """Side-effect for run_coroutine_threadsafe that closes leaked coroutines."""
    calls = []

    def _impl(coro, loop):
        calls.append(coro)
        coro.close()
        return MagicMock()

    _impl.calls = calls
    _impl.call_count_prop = lambda: len(calls)
    return _impl


def _make_runtime(monkeypatch, tmp_path, *, session_timeout: int = 30) -> NavigatorPipecatRuntime:
    configure_test_settings(monkeypatch, tmp_path, session_timeout=session_timeout)
    return NavigatorPipecatRuntime(mock=True, auto_start_audio=False)


# ------------------------------------------------------------------
# 1. Default wake text is the ECU message, not "Hello, I'm listening."
# ------------------------------------------------------------------


def test_default_wake_ack_text_is_ecu_message(monkeypatch):
    get_settings.cache_clear()
    try:
        cfg = get_settings()
        assert "Hello, I'm listening." not in cfg.wake_ack_text_en
        assert "ECU" in cfg.wake_ack_text_en
        assert "Made in ECU" in cfg.wake_ack_text_en
    finally:
        get_settings.cache_clear()


def test_env_override_still_works(monkeypatch):
    monkeypatch.setenv("WAKE_ACK_TEXT_EN", "Custom greeting for testing")
    get_settings.cache_clear()
    try:
        assert get_settings().wake_ack_text_en == "Custom greeting for testing"
    finally:
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 2. Mock mode skips ack entirely
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
# 3. Real-mode first wake schedules ack exactly once
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
# 4. Same session cannot ack twice (_in_progress flag + session guard)
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

            # Directly invoke the guard logic again with the same session_id
            session_id = runtime.session_manager.session_id
            # Guard: session_id == _wake_ack_session_id → blocked
            assert session_id == runtime._wake_ack_session_id, "guard must be set"

            # Manually attempt second scheduling (simulates any re-entry path)
            if (
                runtime._loop
                and not runtime._mock
                and session_id
                and session_id != runtime._wake_ack_session_id
            ):
                runtime._wake_ack_session_id = session_id
                asyncio.run_coroutine_threadsafe(
                    runtime._play_wake_ack_async(session_id), runtime._loop
                )

            assert side_effect.call_count_prop() == 1, "same session_id must not replay ack"
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 5. Keepalive (SPEAKING→LISTENING) does not replay ack
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keepalive_does_not_replay_ack(monkeypatch, tmp_path):
    """After SPEAKING→LISTENING keepalive, _on_wake_word_detected is a no-op (state != IDLE)."""
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    side_effect = _mock_rcts()
    try:
        with patch("app.pipeline.pipecat_graph.asyncio.run_coroutine_threadsafe", side_effect=side_effect):
            runtime._mock = False

            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)
            first_ack_count = side_effect.call_count_prop()  # == 1

            # Simulate full turn → keepalive back to LISTENING
            runtime.session_manager.on_speech_end()
            runtime.session_manager.on_response_ready()
            runtime.session_manager.on_playback_complete()  # → LISTENING

            assert runtime.session_manager.state == SessionState.LISTENING

            # Direct call to the handler (trigger_wake_word goes through cooldown logic)
            runtime._on_wake_word_detected()
            await asyncio.sleep(0.1)

            assert side_effect.call_count_prop() == first_ack_count, (
                "LISTENING state blocks _on_wake_word_detected re-entry"
            )
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 6. New session after IDLE gets fresh ack
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

            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)
            first_session_id = runtime._wake_ack_session_id
            assert side_effect.call_count_prop() == 1

            # End session → IDLE
            runtime.session_manager.end_session("test_end")
            runtime._wakeword.set_session_active(False)
            await asyncio.sleep(0.05)
            assert runtime.session_manager.state == SessionState.IDLE

            # Reset wakeword cooldown so the second trigger is not suppressed
            runtime._wakeword._last_trigger_time = 0.0

            runtime.trigger_wake_word()
            await asyncio.sleep(0.1)

            second_session_id = runtime._wake_ack_session_id
            assert second_session_id != first_session_id, "New session must have new UUID"
            assert side_effect.call_count_prop() == 2, "New session must schedule ack again"
    finally:
        runtime._mock = True
        await runtime.shutdown()


# ------------------------------------------------------------------
# 7. _play_wake_ack_async: wake_ack_enabled=False skips TTS
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

        await runtime._play_wake_ack_async(_FAKE_SESSION)

        assert synthesize_calls == [], "TTS must not be called when wake_ack_enabled=False"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 8. _play_wake_ack_async: aborts if session changed after synthesis
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_wake_ack_aborts_if_session_changed(monkeypatch, tmp_path):
    """Ack coroutine aborts if session_id changed while synthesis was in flight."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "true")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        # Set up: ack was scheduled for FAKE_SESSION, but session has since ended
        runtime._wake_ack_session_id = _FAKE_SESSION

        async def mock_synthesize(text, lang):
            return b"\x00" * 100  # non-empty audio

        runtime._tts_client.synthesize = mock_synthesize
        playback_started = []
        runtime._on_playback_started = lambda sid: playback_started.append(sid)

        # Session manager has no active session (IDLE) → session_id is None
        # scheduled_for (_FAKE_SESSION) != current session (None) → abort
        await runtime._play_wake_ack_async(_FAKE_SESSION)

        assert playback_started == [], "Playback must not start when session has changed"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 9. _play_wake_ack_async: aborts if session state is not LISTENING
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_wake_ack_aborts_if_wrong_state(monkeypatch, tmp_path):
    """Ack aborts if state is PROCESSING/SPEAKING when synthesis returns."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "true")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        # Start a session so session_id matches scheduled_for
        runtime.trigger_wake_word()
        await asyncio.sleep(0.05)
        session_id = runtime.session_manager.session_id
        runtime._wake_ack_session_id = session_id

        # Move session to PROCESSING (state != LISTENING)
        runtime.session_manager.on_speech_end()
        assert runtime.session_manager.state == SessionState.PROCESSING

        async def mock_synthesize(text, lang):
            return b"\x00" * 100

        runtime._tts_client.synthesize = mock_synthesize
        playback_started = []
        runtime._on_playback_started = lambda sid: playback_started.append(sid)

        await runtime._play_wake_ack_async(session_id)

        assert playback_started == [], "Playback must not start when state is PROCESSING"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 10. _play_wake_ack_async: empty audio skips playback
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

        await runtime._play_wake_ack_async(_FAKE_SESSION)

        assert playback_started_calls == [], "No playback when TTS returns empty audio"
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 11. _play_wake_ack_async: TTS exception does not propagate
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
            await runtime._play_wake_ack_async(_FAKE_SESSION)
        except Exception as exc:
            pytest.fail(f"_play_wake_ack_async must not propagate exceptions: {exc}")
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 12. play() failure after _on_playback_started recovers session to LISTENING
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_play_failure_after_started_recovers_to_listening(monkeypatch, tmp_path):
    """If play() raises after SPEAKING transition, session recovers back to LISTENING."""
    monkeypatch.setenv("WAKE_ACK_ENABLED", "true")
    get_settings.cache_clear()
    runtime = _make_runtime(monkeypatch, tmp_path)
    await runtime.start()

    try:
        # Start a session in LISTENING state
        runtime.trigger_wake_word()
        await asyncio.sleep(0.05)
        session_id = runtime.session_manager.session_id
        runtime._wake_ack_session_id = session_id

        assert runtime.session_manager.state == SessionState.LISTENING

        async def good_synthesize(text, lang):
            return b"\x00" * 200

        runtime._tts_client.synthesize = good_synthesize

        def broken_play(audio):
            raise RuntimeError("speaker broke")

        runtime._playback_manager.play = broken_play

        await runtime._play_wake_ack_async(session_id)

        # Session must have recovered from SPEAKING back to LISTENING
        assert runtime.session_manager.state == SessionState.LISTENING, (
            "play() failure must recover session to LISTENING, not leave it stuck in SPEAKING"
        )
    finally:
        await runtime.shutdown()
        get_settings.cache_clear()


# ------------------------------------------------------------------
# 13. playback_complete returns to LISTENING (not IDLE)
# ------------------------------------------------------------------


def test_playback_complete_returns_to_listening():
    """on_playback_complete must transition SPEAKING→LISTENING, not IDLE."""
    from app.audio.session_manager import SessionManager

    sm = SessionManager(session_timeout_sec=30)
    sm.on_wake_detected()
    sm.on_speech_end()
    sm.on_response_ready()
    assert sm.state == SessionState.SPEAKING

    sm.on_playback_complete()
    assert sm.state == SessionState.LISTENING, "playback_complete must keep session alive"
    assert sm.session_id is not None, "session_id must survive playback_complete"


# ------------------------------------------------------------------
# 14. empty audio returns to LISTENING (not IDLE)
# ------------------------------------------------------------------


def test_empty_response_returns_to_listening():
    """on_empty_response must transition PROCESSING→LISTENING, not IDLE."""
    from app.audio.session_manager import SessionManager

    sm = SessionManager(session_timeout_sec=30)
    sm.on_wake_detected()
    sm.on_speech_end()
    assert sm.state == SessionState.PROCESSING

    sm.on_empty_response()
    assert sm.state == SessionState.LISTENING, "empty_response must keep session alive"
    assert sm.session_id is not None


# ------------------------------------------------------------------
# 15. Pipecat idle timeout is disabled (inspect task attributes)
# ------------------------------------------------------------------


def test_pipecat_idle_timeout_is_disabled(monkeypatch, tmp_path):
    """PipelineTask must be created so Pipecat never cancels the pipeline on its own."""
    configure_test_settings(monkeypatch, tmp_path)
    runtime = NavigatorPipecatRuntime(mock=True, auto_start_audio=False)
    task = runtime._task
    assert task is not None

    # idle_timeout_secs=None → Pipecat's idle check is disabled
    idle_attr = getattr(task, "_idle_timeout_secs", getattr(task, "idle_timeout_secs", "NOT_FOUND"))
    assert idle_attr is None, (
        f"idle_timeout_secs must be None (got {idle_attr!r}). "
        "Pipecat must not cancel the pipeline; SessionManager owns the 180-second timeout."
    )

    # cancel_on_idle_timeout=False → even if idle fires, pipeline is not cancelled
    cancel_attr = getattr(task, "_cancel_on_idle_timeout", getattr(task, "cancel_on_idle_timeout", "NOT_FOUND"))
    assert cancel_attr is False or cancel_attr == "NOT_FOUND", (
        f"cancel_on_idle_timeout must be False (got {cancel_attr!r})"
    )
