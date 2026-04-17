"""
Navigator - Pipecat Runtime Graph
Phase 8

Connects the microphone, wake word detector, VAD, STT, controller,
navigation bridge, TTS, and playback manager into one live event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pipecat.frames.frames import (
    CancelFrame,
    DataFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FrameProcessed, FramePushed
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from app.actions.navigation_bridge import NavigationBridge
from app.audio.mic_input import MicCapture
from app.audio.session_manager import SessionManager
from app.pipeline.controller import ConversationController
from app.stt.deepgram_client import DeepgramStreamingClient, load_keyterms_from_db
from app.tts.edge_tts_client import EdgeTTSClient
from app.tts.playback import PlaybackManager
from app.utils.contracts import ResponsePacket, SessionState, TranscriptEvent
from app.utils.logging import get_logger
from app.vad.silero_vad import SileroVAD
from app.wakeword.detector import WakeWordDetector

logger = get_logger(__name__)


class _FallbackGroqClient:
    """Lightweight stand-in used by mock runtime tests."""

    def complete_json(self, *args, **kwargs) -> None:
        return None

    def complete_text(self, *args, **kwargs) -> None:
        return None


@dataclass
class TranscriptEventFrame(DataFrame):
    """Pipeline frame carrying a final transcript event."""

    event: TranscriptEvent


@dataclass
class ResponsePacketFrame(DataFrame):
    """Pipeline frame carrying a controller response packet."""

    packet: ResponsePacket


@dataclass
class SynthesizedAudioFrame(DataFrame):
    """Pipeline frame carrying synthesized TTS bytes."""

    audio: bytes
    language: str = "en"
    session_id: Optional[str] = None
    text: str = ""


@dataclass(frozen=True)
class RuntimeTraceEvent:
    """Structured runtime event for tracing and debugging."""

    name: str
    timestamp: float
    session_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)


class NavigatorRuntimeTracer:
    """Collect runtime events and derive latency metrics per session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[RuntimeTraceEvent] = []
        self._markers: dict[str, dict[str, float]] = {}
        self._latencies: dict[str, dict[str, float]] = {}

    def record(self, name: str, session_id: Optional[str] = None, **data) -> RuntimeTraceEvent:
        timestamp = time.monotonic()
        event = RuntimeTraceEvent(name=name, timestamp=timestamp, session_id=session_id, data=dict(data))

        with self._lock:
            self._events.append(event)
            if session_id:
                markers = self._markers.setdefault(session_id, {})
                markers[name] = timestamp
                self._latencies[session_id] = self._build_latency_snapshot(markers)

        if name == "error_occurred":
            logger.error("runtime_trace", trace_event=name, session_id=session_id, **data)
        else:
            logger.info("runtime_trace", trace_event=name, session_id=session_id, **data)

        return event

    def events(self) -> list[RuntimeTraceEvent]:
        with self._lock:
            return list(self._events)

    def metrics_for(self, session_id: Optional[str]) -> dict[str, float]:
        if not session_id:
            return {}
        with self._lock:
            return dict(self._latencies.get(session_id, {}))

    @staticmethod
    def _build_latency_snapshot(markers: dict[str, float]) -> dict[str, float]:
        def _delta(start: str, end: str) -> Optional[float]:
            if start not in markers or end not in markers:
                return None
            return round(markers[end] - markers[start], 4)

        pairs = {
            "wake_detection_time_sec": ("wake_word_detected", "session_started"),
            "speech_start_time_sec": ("session_started", "speech_started"),
            "speech_end_time_sec": ("session_started", "speech_ended"),
            "stt_final_time_sec": ("session_started", "transcript_final_received"),
            "routing_time_sec": ("transcript_final_received", "intent_decided"),
            "retrieval_time_sec": ("intent_decided", "retrieval_finished"),
            "tts_start_time_sec": ("response_generated", "speaking_started"),
            "tts_end_time_sec": ("speaking_started", "session_ended"),
            "full_turn_latency_sec": ("speech_started", "session_ended"),
        }

        metrics: dict[str, float] = {}
        for label, (start, end) in pairs.items():
            value = _delta(start, end)
            if value is not None:
                metrics[label] = value
        return metrics


class GraphTraceObserver(BaseObserver):
    """Low-level Pipecat frame observer for graph debugging."""

    def __init__(self, max_events: int = 256) -> None:
        super().__init__(name="GraphTraceObserver")
        self._lock = threading.Lock()
        self._max_events = max_events
        self._events: list[dict[str, Any]] = []

    async def on_process_frame(self, data: FrameProcessed):
        self._append(
            kind="process",
            processor=str(data.processor),
            frame=type(data.frame).__name__,
            direction=data.direction.name,
        )

    async def on_push_frame(self, data: FramePushed):
        self._append(
            kind="push",
            source=str(data.source),
            destination=str(data.destination),
            frame=type(data.frame).__name__,
            direction=data.direction.name,
        )

    async def on_pipeline_started(self):
        self._append(kind="pipeline_started")

    async def on_pipeline_finished(self):
        self._append(kind="pipeline_finished")

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def _append(self, **event) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events :]


class DeepgramAdapter(FrameProcessor):
    """Pipecat processor that wraps the Deepgram streaming client."""

    def __init__(self, client: DeepgramStreamingClient, tracer: NavigatorRuntimeTracer) -> None:
        super().__init__(name="DeepgramAdapter")
        self._client = client
        self._tracer = tracer
        self._client._on_partial = self._on_partial  # type: ignore[attr-defined]
        self._client._on_final = self._on_final  # type: ignore[attr-defined]

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._client.set_session_id(session_id)

    def reset_turn(self) -> None:
        self._client.reset_turn()

    def disconnect(self) -> None:
        self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._client.connect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._client.disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            self._client.send_audio(frame.audio)
            return

        await self.push_frame(frame, direction)

    def _on_partial(self, event: TranscriptEvent) -> None:
        logger.debug("pipecat_stt_partial", text=event.text[:80], session_id=event.session_id)

    def _on_final(self, event: TranscriptEvent) -> None:
        self._tracer.record(
            "transcript_final_received",
            session_id=event.session_id,
            text=event.text,
            language=event.language,
        )
        self._schedule_push(TranscriptEventFrame(event=event))

    def _schedule_push(self, frame: Frame) -> None:
        try:
            loop = self.get_event_loop()
        except Exception as exc:
            logger.warning("deepgram_adapter_no_loop", error=str(exc))
            return

        asyncio.run_coroutine_threadsafe(
            self.push_frame(frame, FrameDirection.DOWNSTREAM),
            loop,
        )


class ControllerAdapter(FrameProcessor):
    """Pipecat processor that wraps the conversation controller."""

    def __init__(self, controller: ConversationController) -> None:
        super().__init__(name="ControllerAdapter")
        self._controller = controller

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptEventFrame):
            packet = await asyncio.to_thread(self._controller.handle_transcript, frame.event)
            await self.push_frame(ResponsePacketFrame(packet=packet), FrameDirection.DOWNSTREAM)
            return

        await self.push_frame(frame, direction)


class NavigationAdapter(FrameProcessor):
    """Pipecat processor that emits navigation commands before TTS."""

    def __init__(self, bridge: NavigationBridge, tracer: NavigatorRuntimeTracer) -> None:
        super().__init__(name="NavigationAdapter")
        self._bridge = bridge
        self._tracer = tracer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, ResponsePacketFrame):
            packet = frame.packet

            if packet.should_navigate and packet.navigation_command is not None:
                supplemental = await asyncio.to_thread(
                    self._bridge.navigate,
                    packet.navigation_command,
                    packet.language,
                )
                self._tracer.record(
                    "action_emitted",
                    session_id=packet.session_id,
                    target_code=packet.navigation_command.target_code,
                    target_label=packet.navigation_command.target_label,
                    accepted=supplemental is None,
                )

                if supplemental:
                    packet = ResponsePacket(
                        text=supplemental,
                        language=packet.language,
                        should_navigate=False,
                        navigation_command=None,
                        session_id=packet.session_id,
                    )

                await self.push_frame(ResponsePacketFrame(packet=packet), FrameDirection.DOWNSTREAM)
                return

        await self.push_frame(frame, direction)


class TTSAdapter(FrameProcessor):
    """Pipecat processor that synthesizes response text to audio."""

    def __init__(self, tts: EdgeTTSClient, tracer: NavigatorRuntimeTracer) -> None:
        super().__init__(name="TTSAdapter")
        self._tts = tts
        self._tracer = tracer

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, ResponsePacketFrame):
            packet = frame.packet
            audio = await self._tts.synthesize(packet.text, packet.language)
            if not audio:
                self._tracer.record(
                    "error_occurred",
                    session_id=packet.session_id,
                    source="tts",
                    message="synthesis_produced_no_audio",
                )

            await self.push_frame(
                SynthesizedAudioFrame(
                    audio=audio,
                    language=packet.language,
                    session_id=packet.session_id,
                    text=packet.text,
                ),
                FrameDirection.DOWNSTREAM,
            )
            return

        await self.push_frame(frame, direction)


class PlaybackAdapter(FrameProcessor):
    """Pipecat processor that hands synthesized audio to the speaker."""

    def __init__(
        self,
        playback: PlaybackManager,
        tracer: NavigatorRuntimeTracer,
        on_playback_started,
        on_empty_audio,
    ) -> None:
        super().__init__(name="PlaybackAdapter")
        self._playback = playback
        self._tracer = tracer
        self._on_playback_started = on_playback_started
        self._on_empty_audio = on_empty_audio

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, SynthesizedAudioFrame):
            if not frame.audio:
                self._on_empty_audio(frame.session_id)
                return

            self._on_playback_started(frame.session_id)
            self._tracer.record(
                "speaking_started",
                session_id=frame.session_id,
                text=frame.text[:120],
                language=frame.language,
            )
            self._playback.play(frame.audio)
            return

        if isinstance(frame, InterruptionFrame):
            self._playback.notify_speech_detected()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._playback.stop()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)


class NavigatorPipecatRuntime:
    """Live Pipecat runtime that wires the full Navigator voice loop."""

    def __init__(
        self,
        *,
        mock: bool = False,
        auto_start_audio: bool = True,
        mic: Optional[MicCapture] = None,
        wakeword: Optional[WakeWordDetector] = None,
        vad: Optional[SileroVAD] = None,
        stt_client: Optional[DeepgramStreamingClient] = None,
        controller: Optional[ConversationController] = None,
        tts_client: Optional[EdgeTTSClient] = None,
        playback_manager: Optional[PlaybackManager] = None,
        navigation_bridge: Optional[NavigationBridge] = None,
        session_manager: Optional[SessionManager] = None,
    ) -> None:
        self._mock = mock
        self._auto_start_audio = auto_start_audio
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[PipelineRunner] = None
        self._runner_task: Optional[asyncio.Task] = None
        self._task: Optional[PipelineTask] = None
        self._pipeline_started = asyncio.Event()
        self._pipeline_finished = asyncio.Event()
        self._stop_requested = asyncio.Event()
        self._running = False
        self._mic_thread: Optional[threading.Thread] = None
        self._last_session_id: Optional[str] = None

        self._tracer = NavigatorRuntimeTracer()
        self._observer = GraphTraceObserver()

        keyterms = [] if mock else load_keyterms_from_db()
        fallback_groq = _FallbackGroqClient()

        self._session_manager = session_manager or SessionManager(on_timeout=self._on_session_timeout)
        self._mic = mic or MicCapture(mock=mock)
        self._wakeword = wakeword or WakeWordDetector(mock=mock)
        self._vad = vad or SileroVAD(mock=mock)
        self._stt_client = stt_client or DeepgramStreamingClient(mock=mock, keyterms=keyterms)
        self._controller = controller or ConversationController(groq=fallback_groq if mock else None)
        self._tts_client = tts_client or EdgeTTSClient(mock=mock)
        self._playback_manager = playback_manager or PlaybackManager(mock=mock)
        self._navigation_bridge = navigation_bridge or NavigationBridge(mock=mock)

        self._controller.set_trace_hook(self._tracer.record)
        self._bind_callbacks()

        self._deepgram_adapter = DeepgramAdapter(self._stt_client, self._tracer)
        self._controller_adapter = ControllerAdapter(self._controller)
        self._navigation_adapter = NavigationAdapter(self._navigation_bridge, self._tracer)
        self._tts_adapter = TTSAdapter(self._tts_client, self._tracer)
        self._playback_adapter = PlaybackAdapter(
            self._playback_manager,
            self._tracer,
            on_playback_started=self._on_playback_started,
            on_empty_audio=self._on_empty_audio,
        )

        pipeline = Pipeline(
            [
                self._deepgram_adapter,
                self._controller_adapter,
                self._navigation_adapter,
                self._tts_adapter,
                self._playback_adapter,
            ]
        )
        params = PipelineParams(
            audio_in_sample_rate=self._mic.sample_rate,
            enable_metrics=True,
            start_metadata={"component": "navigator_runtime"},
        )
        self._task = PipelineTask(pipeline, params=params, observers=[self._observer])

        @self._task.event_handler("on_pipeline_started")
        async def _on_pipeline_started(*_args, **_kwargs):
            self._pipeline_started.set()
            logger.info("pipecat_pipeline_started")

        @self._task.event_handler("on_pipeline_finished")
        async def _on_pipeline_finished(*_args, **_kwargs):
            self._pipeline_finished.set()
            logger.info("pipecat_pipeline_finished")

        @self._task.event_handler("on_pipeline_error")
        async def _on_pipeline_error(_task, error):
            self._tracer.record(
                "error_occurred",
                session_id=self._session_manager.session_id or self._last_session_id,
                source="pipecat_pipeline",
                message=str(error),
            )

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def tracer(self) -> NavigatorRuntimeTracer:
        return self._tracer

    @property
    def observer(self) -> GraphTraceObserver:
        return self._observer

    @property
    def vad(self) -> SileroVAD:
        return self._vad

    @property
    def playback_manager(self) -> PlaybackManager:
        return self._playback_manager

    async def start(self) -> None:
        """Start the Pipecat task and, optionally, the microphone loop."""
        if self._running:
            return

        self._loop = asyncio.get_running_loop()
        self._runner = PipelineRunner(name="navigator-runtime", handle_sigint=False)
        self._runner_task = asyncio.create_task(self._runner.run(self._task))
        await asyncio.wait_for(self._pipeline_started.wait(), timeout=5.0)

        self._running = True
        if self._auto_start_audio:
            self._mic.start()
            self._mic_thread = threading.Thread(
                target=self._mic_loop,
                daemon=True,
                name="navigator-mic-loop",
            )
            self._mic_thread.start()

        logger.info("navigator_runtime_started", mock=self._mock, auto_start_audio=self._auto_start_audio)

    async def run(self) -> None:
        """Run until shutdown is requested."""
        await self.start()
        try:
            await self._stop_requested.wait()
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Stop the mic loop, drain playback, and close the Pipecat task."""
        if self._runner is None or self._task is None:
            return

        if not self._stop_requested.is_set():
            self._stop_requested.set()

        self._running = False
        self._mic.stop()
        if self._mic_thread:
            self._mic_thread.join(timeout=2.0)
            self._mic_thread = None

        self._playback_manager.stop()
        self._wakeword.stop()
        self._wakeword.set_session_active(False)
        self._vad.reset()
        self._deepgram_adapter.set_session_id(None)

        if not self._pipeline_finished.is_set():
            try:
                await self._task.queue_frame(EndFrame(reason="runtime_shutdown"))
            except Exception as exc:
                logger.warning("runtime_end_frame_failed", error=str(exc))

        if self._runner_task:
            try:
                await asyncio.wait_for(self._runner_task, timeout=5.0)
            except asyncio.TimeoutError:
                await self._runner.cancel()
                await self._runner_task

        self._deepgram_adapter.disconnect()
        logger.info("navigator_runtime_stopped")

    async def wait_for_state(self, state: SessionState, timeout: float = 5.0) -> bool:
        """Wait until the session manager reaches the requested state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._session_manager.state == state:
                return True
            await asyncio.sleep(0.02)
        return self._session_manager.state == state

    def trigger_wake_word(self) -> None:
        """Public helper for tests and manual runtime control."""
        self._wakeword.trigger()

    def inject_mock_transcript(self, text: str, *, is_final: bool = True, language: str = "en") -> None:
        """Inject a mock transcript through the wrapped STT client."""
        self._stt_client._language = language  # type: ignore[attr-defined]
        self._deepgram_adapter.set_session_id(self._session_manager.session_id or self._last_session_id)
        self._stt_client.inject_mock_transcript(text, is_final=is_final)

    def process_audio_frame(self, frame: bytes) -> None:
        """Feed one microphone frame into the live session state machine."""
        state = self._session_manager.state

        if state == SessionState.IDLE:
            self._wakeword.process_frame(frame)
            return

        if state in (
            SessionState.WAKE_DETECTED,
            SessionState.LISTENING,
            SessionState.INTERRUPTED,
            SessionState.SPEAKING,
        ):
            self._vad.process(frame)

    def _bind_callbacks(self) -> None:
        self._wakeword._on_activated = self._on_wake_word_detected  # type: ignore[attr-defined]
        self._vad._on_speech_start = self._on_speech_start  # type: ignore[attr-defined]
        self._vad._on_speech_end = self._on_speech_end  # type: ignore[attr-defined]
        self._vad._on_speech_frame = self._on_speech_frame  # type: ignore[attr-defined]
        self._playback_manager._on_complete = self._on_playback_complete  # type: ignore[attr-defined]
        self._session_manager._on_timeout = self._on_session_timeout  # type: ignore[attr-defined]

    def _mic_loop(self) -> None:
        for frame in self._mic.frames():
            if not self._running:
                break
            try:
                self.process_audio_frame(frame)
            except Exception as exc:
                logger.error("runtime_mic_loop_error", error=str(exc))
                self._tracer.record(
                    "error_occurred",
                    session_id=self._session_manager.session_id or self._last_session_id,
                    source="mic_loop",
                    message=str(exc),
                )

    def _on_wake_word_detected(self) -> None:
        if self._session_manager.state != SessionState.IDLE:
            return

        self._tracer.record("wake_word_detected")
        self._session_manager.on_wake_detected()
        session_id = self._session_manager.session_id
        self._last_session_id = session_id

        self._wakeword.set_session_active(True)
        self._vad.reset()
        self._deepgram_adapter.set_session_id(session_id)
        self._deepgram_adapter.reset_turn()
        self._session_manager.start_timeout_timer()
        self._tracer.record("session_started", session_id=session_id)

    def _on_speech_start(self) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        if self._session_manager.state == SessionState.SPEAKING:
            self._session_manager.on_barge_in()
            self._tracer.record("speaking_interrupted", session_id=session_id)
            self._queue_frame_threadsafe(InterruptionFrame())
        else:
            self._session_manager.on_speech_start()

        self._session_manager.activity_ping()
        self._deepgram_adapter.reset_turn()
        self._queue_frame_threadsafe(UserStartedSpeakingFrame())
        self._tracer.record("speech_started", session_id=session_id)

    def _on_speech_frame(self, frame: bytes) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        self._session_manager.activity_ping()
        self._queue_frame_threadsafe(InputAudioRawFrame(frame, self._mic.sample_rate, 1))

    def _on_speech_end(self) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        self._session_manager.on_speech_end()
        self._queue_frame_threadsafe(UserStoppedSpeakingFrame())
        self._tracer.record("speech_ended", session_id=session_id)

    def _on_playback_started(self, session_id: Optional[str]) -> None:
        if session_id is None:
            session_id = self._session_manager.session_id or self._last_session_id
        self._session_manager.on_response_ready()
        self._session_manager.activity_ping()

    def _on_empty_audio(self, session_id: Optional[str]) -> None:
        if session_id is None:
            session_id = self._session_manager.session_id or self._last_session_id

        self._tracer.record(
            "error_occurred",
            session_id=session_id,
            source="playback",
            message="no_audio_available_for_playback",
        )
        self._session_manager.reset()
        self._wakeword.set_session_active(False)
        self._deepgram_adapter.set_session_id(None)
        self._tracer.record("session_ended", session_id=session_id, reason="empty_audio")

    def _on_playback_complete(self) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        self._session_manager.on_playback_complete()
        self._wakeword.set_session_active(False)
        self._deepgram_adapter.set_session_id(None)
        self._vad.reset()
        self._tracer.record("session_ended", session_id=session_id, reason="playback_complete")

    def _on_session_timeout(self) -> None:
        session_id = self._last_session_id
        self._wakeword.set_session_active(False)
        self._deepgram_adapter.set_session_id(None)
        self._vad.reset()
        self._tracer.record("session_ended", session_id=session_id, reason="timeout")

    def _queue_frame_threadsafe(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        if self._loop is None or self._task is None or self._loop.is_closed():
            return

        asyncio.run_coroutine_threadsafe(
            self._task.queue_frame(frame, direction=direction),
            self._loop,
        )
