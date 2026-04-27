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
from app.config import get_settings
from app.pipeline.controller import ConversationController
from app.stt.deepgram_client import DeepgramStreamingClient
from app.stt.dual_stt_client import DualSTTClient
from app.tts.edge_tts_client import EdgeTTSClient
from app.tts.playback import PlaybackManager
from app.ui.mqtt_publisher import MQTTPublisher
from app.ui.status_publisher import StatusPublisher
from app.utils.contracts import ResponsePacket, SessionState, TranscriptEvent
from app.utils.logging import get_logger
from app.vad.silero_vad import SileroVAD
from app.wakeword.detector import WakeWordDetector

logger = get_logger(__name__)
_INFO_TRACE_EVENTS = {
    "wake_word_detected",
    "session_started",
    "speech_started",
    "speech_ended",
    "deepgram_connected",
    "transcript_final_received",
    "intent_decided",
    "retrieval_finished",
    "response_generated",
    "speaking_started",
    "speaking_interrupted",
    "session_ended",
}


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
        elif name == "tts_empty_audio":
            logger.warning("runtime_trace", trace_event=name, session_id=session_id, **data)
        elif name in _INFO_TRACE_EVENTS:
            logger.info("runtime_trace", trace_event=name, session_id=session_id, **data)
        else:
            logger.debug("runtime_trace", trace_event=name, session_id=session_id, **data)

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
        self._client.set_callbacks(
            on_partial=self._on_partial,
            on_final=self._on_final,
            on_connected=self._on_connected,
            on_error=self._on_error,
        )

    def connect(self) -> None:
        self._client.connect()

    def set_session_id(self, session_id: Optional[str]) -> None:
        self._client.set_session_id(session_id)

    def reset_turn(self) -> None:
        self._client.reset_turn()

    def disconnect(self) -> None:
        self._client.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (EndFrame, CancelFrame)):
            self._client.disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._client.finalize_turn()
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

    def _on_connected(self) -> None:
        self._tracer.record(
            "deepgram_connected",
            session_id=self._client.session_id,
        )

    def _on_error(self, reason: str, message: str) -> None:
        self._tracer.record(
            "error_occurred",
            session_id=self._client.session_id,
            source="deepgram",
            reason=reason,
            message=message,
        )

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
            started = time.monotonic()
            audio = await self._tts.synthesize(packet.text, packet.language)
            elapsed_ms = (time.monotonic() - started) * 1000
            if elapsed_ms > 400:
                logger.warning(
                    "controller.latency_budget_exceeded",
                    stage="tts",
                    budget_ms=400,
                    actual_ms=round(elapsed_ms),
                )
            if not audio:
                self._tracer.record(
                    "tts_empty_audio",
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
        self._session_end_lock = threading.Lock()
        # Guard: track which session_id already received the wake acknowledgment.
        # Each new session has a fresh UUID, so the comparison naturally prevents
        # replaying the ack within the same session or during keepalive transitions.
        # The state machine (_on_wake_word_detected returns if state != IDLE) prevents
        # re-entry within the same session, so no in-progress flag is needed.
        self._wake_ack_session_id: Optional[str] = None

        self._tracer = NavigatorRuntimeTracer()
        self._observer = GraphTraceObserver()

        cfg = get_settings()
        self._status_publisher = StatusPublisher(
            json_path=cfg.status_json_path,
            ws_enabled=cfg.status_ws_enabled,
            ws_host=cfg.status_ws_host,
            ws_port=cfg.status_ws_port,
        )
        self._mqtt = MQTTPublisher(
            enabled=cfg.mqtt_enabled,
            broker=cfg.mqtt_broker,
            port=cfg.mqtt_port,
            username=cfg.mqtt_username,
            password=cfg.mqtt_password,
            topic=cfg.mqtt_topic,
            tls_enabled=cfg.mqtt_tls_enabled,
            qos=cfg.mqtt_qos,
            retain=cfg.mqtt_retain,
        )

        fallback_groq = _FallbackGroqClient()

        self._session_manager = session_manager or SessionManager(
            session_timeout_sec=cfg.session_timeout_sec,
            on_timeout=self._on_session_timeout,
            on_reset=self._on_session_reset,
        )
        self._mic = mic or MicCapture(mock=mock)
        self._wakeword = wakeword or WakeWordDetector(mock=mock)
        self._vad = vad or SileroVAD(mock=mock)
        self._stt_client = stt_client or DualSTTClient(mock=mock)
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
        # Disable Pipecat's internal idle-timeout. Our SessionManager owns the
        # 180-second conversation timeout; Pipecat must never cancel the pipeline.
        self._task = PipelineTask(
            pipeline,
            params=params,
            observers=[self._observer],
            idle_timeout_secs=None,
            cancel_on_idle_timeout=False,
        )

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
        self._status_publisher.start()
        self._mqtt.start()
        self._status_publisher.publish(
            event="idle",
            state="idle",
            message="Waiting for wake word",
            is_listening=False,
            is_speaking=False,
            wake_word_detected=False,
        )
        self._mqtt.publish_state("idle")

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
        self._status_publisher.stop()
        self._mqtt.stop()
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

    def inject_mock_transcript(
        self,
        text: str,
        *,
        is_final: bool = True,
        language: str = "en",
        language_confidence: float | None = None,
    ) -> None:
        """Inject a mock transcript through the wrapped STT client."""
        self._deepgram_adapter.set_session_id(self._session_manager.session_id or self._last_session_id)
        self._stt_client.inject_mock_transcript(
            text,
            is_final=is_final,
            language=language,
            language_confidence=language_confidence,
        )

    def process_audio_frame(self, frame: bytes) -> None:
        """Feed one microphone frame into the live session state machine."""
        state = self._session_manager.state

        if state == SessionState.IDLE:
            self._wakeword.process_frame(frame)
            return

        if state in (
            SessionState.WAKE_DETECTED,
            SessionState.LISTENING,
            SessionState.SPEAKING,
            SessionState.INTERRUPTED,
        ):
            self._vad.process(frame)

    def _bind_callbacks(self) -> None:
        self._wakeword._on_activated = self._on_wake_word_detected  # type: ignore[attr-defined]
        self._vad._on_speech_start = self._on_speech_start  # type: ignore[attr-defined]
        self._vad._on_speech_end = self._on_speech_end  # type: ignore[attr-defined]
        self._vad._on_speech_frame = self._on_speech_frame  # type: ignore[attr-defined]
        self._playback_manager._on_complete = self._on_playback_complete  # type: ignore[attr-defined]
        self._session_manager._on_timeout = self._on_session_timeout  # type: ignore[attr-defined]
        self._session_manager._on_reset = self._on_session_reset  # type: ignore[attr-defined]

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
        self._deepgram_adapter.connect()

        self._status_publisher.publish(
            event="listening",
            state="listening",
            message="Listening...",
            session_id=session_id,
            is_listening=True,
            is_speaking=False,
            wake_word_detected=True,
        )
        self._mqtt.publish_state("listening")

        # Play wake acknowledgment once per unique session (real mode only).
        # Guard conditions:
        #   - session_id must be non-None (race: session might not be set yet)
        #   - session_id must differ from the last acked session
        #   - no ack synthesis must currently be in flight
        # _wake_ack_session_id is set BEFORE scheduling to block any concurrent trigger.
        if (
            self._loop
            and not self._mock
            and session_id
            and session_id != self._wake_ack_session_id
        ):
            self._wake_ack_session_id = session_id
            asyncio.run_coroutine_threadsafe(
                self._play_wake_ack_async(session_id), self._loop
            )

    def _on_speech_start(self) -> None:
        if self._playback_manager.is_echo_suppressed():
            logger.debug("runtime.echo_suppressed_speech_start_ignored")
            return

        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        if self._session_manager.state == SessionState.SPEAKING:
            if not self._playback_manager.notify_speech_detected():
                return
            self._session_manager.on_barge_in()
            self._tracer.record("speaking_interrupted", session_id=session_id)
            self._deepgram_adapter.connect()
            self._deepgram_adapter.reset_turn()
            self._queue_frame_threadsafe(InterruptionFrame())
            self._mqtt.publish_state("listening")
            return

        self._session_manager.on_speech_start()

        self._session_manager.activity_ping()
        self._deepgram_adapter.connect()
        self._deepgram_adapter.reset_turn()
        self._queue_frame_threadsafe(UserStartedSpeakingFrame())
        self._tracer.record("speech_started", session_id=session_id)
        self._status_publisher.publish(
            event="listening",
            state="listening",
            message="Listening...",
            session_id=session_id,
            is_listening=True,
            is_speaking=False,
            wake_word_detected=True,
        )

    def _on_speech_frame(self, frame: bytes) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        if self._session_manager.state == SessionState.SPEAKING:
            logger.debug("runtime.speech_frame_ignored_during_speaking")
            return

        self._session_manager.activity_ping()
        self._queue_frame_threadsafe(InputAudioRawFrame(frame, self._mic.sample_rate, 1))

    def _on_speech_end(self) -> None:
        session_id = self._session_manager.session_id or self._last_session_id
        if not session_id:
            return

        if self._session_manager.state == SessionState.SPEAKING:
            logger.debug("runtime.speech_end_ignored_during_speaking")
            return

        self._session_manager.on_speech_end()
        self._queue_frame_threadsafe(UserStoppedSpeakingFrame())
        self._tracer.record("speech_ended", session_id=session_id)
        self._status_publisher.publish(
            event="processing",
            state="processing",
            message="Processing...",
            session_id=session_id,
            is_listening=False,
            is_speaking=False,
            wake_word_detected=True,
        )
        self._mqtt.publish_state("processing")

    def _on_playback_started(self, session_id: Optional[str]) -> None:
        if session_id is None:
            session_id = self._session_manager.session_id or self._last_session_id
        self._session_manager.on_response_ready()
        self._status_publisher.publish(
            event="speaking",
            state="speaking",
            message="Speaking...",
            session_id=session_id,
            is_listening=False,
            is_speaking=True,
            wake_word_detected=True,
        )
        self._mqtt.publish_state("speaking")

    def _on_empty_audio(self, session_id: Optional[str]) -> None:
        if session_id is None:
            session_id = self._session_manager.session_id or self._last_session_id

        self._tracer.record(
            "tts_empty_audio",
            session_id=session_id,
            source="playback",
            message="empty_audio_keepalive",
        )
        logger.warning("tts_empty_audio_returning_to_listening", session_id=session_id)
        self._session_manager.on_empty_response()
        self._deepgram_adapter.set_session_id(session_id)
        self._deepgram_adapter.reset_turn()
        self._status_publisher.publish(
            event="listening",
            state="listening",
            message="Listening...",
            session_id=session_id,
            is_listening=True,
            is_speaking=False,
            wake_word_detected=True,
        )
        self._mqtt.publish_state("listening")

    def _on_playback_complete(self) -> None:
        """TTS finished naturally — keep the session alive and return to LISTENING."""
        self._session_manager.on_playback_complete()
        session_id = self._session_manager.session_id or self._last_session_id
        # Prepare deepgram for the next utterance in this session
        self._deepgram_adapter.set_session_id(session_id)
        self._deepgram_adapter.reset_turn()
        self._tracer.record("playback_complete_keepalive", session_id=session_id)
        self._status_publisher.publish(
            event="listening",
            state="listening",
            message="Listening...",
            session_id=session_id,
            is_listening=True,
            is_speaking=False,
            wake_word_detected=True,
        )
        self._mqtt.publish_state("listening")

    def _on_session_reset(self) -> None:
        """
        Re-arm listening after the session manager auto-recovers from ERROR.
        """
        logger.info("runtime.session_reset_to_idle_after_error")
        self._wakeword.set_session_active(False)
        self._deepgram_adapter.disconnect()
        self._deepgram_adapter.reset_turn()
        self._deepgram_adapter.set_session_id(None)
        self._vad.reset()
        self._tracer.record("session_reset_complete")

    def _on_session_timeout(self, reason: str = "timeout") -> None:
        self._on_session_ended(reason=reason)

    def _on_session_ended(
        self,
        reason: str,
        session_id: Optional[str] = None,
        *,
        stop_playback: bool = True,
    ) -> None:
        with self._session_end_lock:
            active_session_id = session_id or self._session_manager.session_id or self._last_session_id
            if active_session_id is None and self._session_manager.state == SessionState.IDLE:
                return

            logger.info(
                "runtime.session_ending",
                reason=reason,
                session_id=active_session_id,
                state=self._session_manager.state.value,
            )
            if stop_playback:
                self._playback_manager.stop()
            self._wakeword.set_session_active(False)
            self._deepgram_adapter.disconnect()
            self._deepgram_adapter.reset_turn()
            self._deepgram_adapter.set_session_id(None)
            self._vad.reset()

            ended_session_id = self._session_manager.end_session(reason=reason) or active_session_id
            if ended_session_id:
                self._last_session_id = ended_session_id

            self._tracer.record("session_ended", session_id=ended_session_id, reason=reason)
            event_name = "session_timeout" if reason == "timeout" else "session_ended"
            self._status_publisher.publish(
                event=event_name,
                state="idle",
                message="Conversation ended due to inactivity" if reason == "timeout" else "Conversation ended",
                session_id=ended_session_id,
                is_listening=False,
                is_speaking=False,
                wake_word_detected=False,
            )
            self._mqtt.publish_state("idle")

    async def _play_wake_ack_async(self, scheduled_for: str) -> None:
        """Synthesize and play the wake acknowledgment phrase (real mode only).

        Args:
            scheduled_for: The session UUID that triggered this ack.  If the
                session has changed by the time synthesis completes (because the
                session ended or a new wake arrived), the ack is discarded so
                only one greeting plays per session.
        """
        cfg = get_settings()
        if not cfg.wake_ack_enabled:
            self._wake_ack_in_progress = False
            return

        started_playback = False
        try:
            audio = await self._tts_client.synthesize(cfg.wake_ack_text_en, cfg.wake_ack_language)
            if not audio:
                return

            # Synthesis is async — by the time it returns the session may have
            # changed.  Only play if this coroutine was scheduled for the session
            # that is still active AND the session is in a state where playing
            # a greeting makes sense (LISTENING or WAKE_DETECTED).
            current_session_id = self._session_manager.session_id
            if current_session_id != scheduled_for:
                logger.debug(
                    "wake_ack_aborted_session_changed",
                    scheduled_for=scheduled_for,
                    current=current_session_id,
                )
                return

            current_state = self._session_manager.state
            if current_state not in (SessionState.LISTENING, SessionState.WAKE_DETECTED):
                logger.debug(
                    "wake_ack_skipped_wrong_state",
                    state=current_state.value,
                    session_id=scheduled_for,
                )
                return

            self._on_playback_started(scheduled_for)
            started_playback = True
            self._playback_manager.play(audio)

        except Exception as exc:
            logger.warning("wake_ack_failed", error=str(exc))
            if started_playback:
                # _on_playback_started transitioned the session to SPEAKING and
                # cancelled the inactivity timer.  play() failed before any audio
                # was queued so _on_playback_complete will never fire.  Recover
                # by returning the session to LISTENING with a fresh timer.
                self._session_manager.on_empty_response()

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
