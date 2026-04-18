"""
Navigator - Session State Machine
Phase 3, Step 3.4

Manages the lifecycle of one conversation turn.
Every state transition is logged and explicit — no silent state changes.

States:
    IDLE          -> waiting for wake word
    WAKE_DETECTED -> wake word fired, opening session
    LISTENING     -> VAD active, capturing user speech
    PROCESSING    -> STT final received, routing in progress
    SPEAKING      -> TTS playback active
    INTERRUPTED   -> user spoke during TTS playback (barge-in)
    ERROR         -> unrecoverable error in current turn

Allowed transitions:
    IDLE          -> WAKE_DETECTED
    WAKE_DETECTED -> LISTENING
    LISTENING     -> PROCESSING
    LISTENING     -> IDLE          (timeout)
    PROCESSING    -> SPEAKING
    PROCESSING    -> IDLE          (unknown / no response)
    SPEAKING      -> IDLE          (playback complete)
    SPEAKING      -> INTERRUPTED   (barge-in)
    INTERRUPTED   -> LISTENING     (resume after barge-in)
    ERROR         -> IDLE          (reset after error)
    Any           -> ERROR
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from app.utils.contracts import SessionState
from app.utils.logging import get_logger

logger = get_logger(__name__)


# Adjacency map: valid transitions
_ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.IDLE:          {SessionState.WAKE_DETECTED},
    SessionState.WAKE_DETECTED: {SessionState.LISTENING, SessionState.ERROR},
    SessionState.LISTENING:     {SessionState.PROCESSING, SessionState.IDLE, SessionState.ERROR},
    SessionState.PROCESSING:    {SessionState.SPEAKING, SessionState.IDLE, SessionState.ERROR},
    SessionState.SPEAKING:      {SessionState.IDLE, SessionState.INTERRUPTED, SessionState.ERROR},
    SessionState.INTERRUPTED:   {SessionState.LISTENING, SessionState.IDLE, SessionState.ERROR},
    SessionState.ERROR:         {SessionState.IDLE},
}


class SessionManager:
    """
    Manages Navigator conversation session state.

    Thread-safe: transitions can be triggered from the audio thread,
    the STT callback thread, and the TTS playback thread simultaneously.

    Args:
        session_timeout_sec: Seconds before an idle LISTENING state times out.
        on_timeout:          Optional callback when session times out.
    """

    def __init__(
        self,
        session_timeout_sec: Optional[int] = None,
        on_timeout: Optional[callable] = None,
    ) -> None:
        from app.config import get_settings
        cfg = get_settings()

        self._timeout_sec   = session_timeout_sec or cfg.session_timeout_sec
        self._on_timeout    = on_timeout
        self._state         = SessionState.IDLE
        self._session_id: Optional[str] = None
        self._last_activity = time.monotonic()
        self._lock          = threading.Lock()
        self._timer: Optional[threading.Timer] = None

        logger.info("session_manager_init", timeout_sec=self._timeout_sec)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        """Current session state (thread-safe read)."""
        with self._lock:
            return self._state

    @property
    def session_id(self) -> Optional[str]:
        """Current session UUID, or None in IDLE state."""
        with self._lock:
            return self._session_id

    def transition(self, new_state: SessionState) -> bool:
        """
        Attempt to move to a new state.

        Returns True if the transition was accepted, False if it was illegal.
        Error transitions are always allowed from any state.
        """
        with self._lock:
            current = self._state

            # ERROR is always reachable from any state
            if new_state == SessionState.ERROR:
                return self._apply(current, new_state)

            allowed = _ALLOWED_TRANSITIONS.get(current, set())
            if new_state not in allowed:
                logger.warning(
                    "session_invalid_transition",
                    from_state=current.value,
                    to_state=new_state.value,
                )
                return False

            return self._apply(current, new_state)

    def reset(self) -> None:
        """Force the session back to IDLE and clear the session ID."""
        with self._lock:
            self._cancel_timer()
            old = self._state
            self._state = SessionState.IDLE
            self._session_id = None
            logger.info("session_reset", from_state=old.value)

    # ── State helpers ─────────────────────────────────────────────────────────

    def on_wake_detected(self) -> None:
        """Called when the wake word fires. Opens a new session."""
        with self._lock:
            self._cancel_timer()
            current = self._state
            if current != SessionState.IDLE:
                logger.warning("session_wake_ignored_not_idle", state=current.value)
                return
            self._session_id = str(uuid.uuid4())
            self._apply(current, SessionState.WAKE_DETECTED)
        self.transition(SessionState.LISTENING)

    def on_speech_start(self) -> None:
        """Called when VAD detects speech onset."""
        self._last_activity = time.monotonic()
        current = self.state
        if current in (SessionState.WAKE_DETECTED, SessionState.INTERRUPTED):
            self.transition(SessionState.LISTENING)

    def on_speech_end(self) -> None:
        """Called when VAD detects end of utterance — start processing."""
        self.transition(SessionState.PROCESSING)

    def on_response_ready(self) -> None:
        """Called when a spoken response is ready to play."""
        self.transition(SessionState.SPEAKING)

    def on_playback_complete(self) -> None:
        """Called when TTS finishes speaking."""
        self.transition(SessionState.IDLE)
        with self._lock:
            self._session_id = None

    def on_barge_in(self) -> None:
        """Called when user speaks during TTS — barge-in fires."""
        logger.info("session_barge_in", session_id=self._session_id)
        self.transition(SessionState.INTERRUPTED)
        self.transition(SessionState.LISTENING)

    def on_error(self, reason: str = "unknown") -> None:
        """Mark the session as errored and schedule a reset."""
        logger.error("session_error", reason=reason, session_id=self._session_id)
        self.transition(SessionState.ERROR)
        self._schedule_reset()

    def start_timeout_timer(self) -> None:
        """Start the inactivity timer. Call when entering LISTENING state."""
        with self._lock:
            self._cancel_timer()
            self._timer = threading.Timer(
                interval=self._timeout_sec,
                function=self._handle_timeout,
            )
            self._timer.daemon = True
            self._timer.start()

    def activity_ping(self) -> None:
        """Reset the inactivity timer (call when speech or input is detected)."""
        self._last_activity = time.monotonic()
        with self._lock:
            self._cancel_timer()
        self.start_timeout_timer()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _apply(self, old: SessionState, new: SessionState) -> bool:
        """Write the new state and log it. Must be called with lock held."""
        self._state = new
        logger.info(
            "session_transition",
            from_state=old.value,
            to_state=new.value,
            session_id=self._session_id,
        )
        return True

    def _cancel_timer(self) -> None:
        """Cancel the running inactivity timer. Must be called with lock held."""
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _handle_timeout(self) -> None:
        """Called by the timer thread when the session times out."""
        with self._lock:
            current = self._state
        if current in (SessionState.LISTENING, SessionState.WAKE_DETECTED):
            logger.info("session_timeout", state=current.value, session_id=self._session_id)
            self.transition(SessionState.IDLE)
            with self._lock:
                self._session_id = None
            if self._on_timeout:
                try:
                    self._on_timeout()
                except Exception as exc:
                    logger.error("session_timeout_callback_error", error=str(exc))

    def _schedule_reset(self) -> None:
        """Schedule an automatic reset from ERROR state after 1 second."""
        def _do_reset():
            time.sleep(1.0)
            self.reset()
        t = threading.Thread(target=_do_reset, daemon=True)
        t.start()
