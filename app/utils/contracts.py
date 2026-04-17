"""
Navigator - Internal Event Contracts
Typed models for all data passed between modules.

Rules:
- Every module boundary must use these models, never raw dicts.
- Models are immutable by default (frozen=True where safe).
- All fields have explicit types and docstring descriptions.
- Optional fields use None as default, not empty strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

class SessionState(str, Enum):
    """
    All valid states of a Navigator conversation session.
    Transitions must be explicit and logged.
    """
    IDLE         = "idle"           # Waiting for wake word
    WAKE_DETECTED = "wake_detected" # Wake word fired, opening session
    LISTENING    = "listening"      # VAD active, capturing user speech
    PROCESSING   = "processing"     # STT final received, routing in progress
    SPEAKING     = "speaking"       # TTS playback active
    INTERRUPTED  = "interrupted"    # User spoke during TTS — barge-in fired
    ERROR        = "error"          # Unrecoverable error in current turn


# ─────────────────────────────────────────────────────────────────────────────
# Intent Classes
# ─────────────────────────────────────────────────────────────────────────────

class IntentClass(str, Enum):
    """
    All valid intent classes the router may return.
    These are the only valid routing targets in the MVP.
    """
    CAMPUS_QUERY       = "campus_query"
    NAVIGATION_REQUEST = "navigation_request"
    SOCIAL_CHAT        = "social_chat"
    UNKNOWN            = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Status
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalStatus(str, Enum):
    """Outcome codes from the truth layer retrieval engine."""
    OK          = "ok"          # One strong match found
    AMBIGUOUS   = "ambiguous"   # Multiple candidates, needs clarification
    NOT_FOUND   = "not_found"   # No trustworthy match in the database
    ERROR       = "error"       # Internal retrieval failure


# ─────────────────────────────────────────────────────────────────────────────
# Transcript Event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptEvent:
    """
    Produced by the STT module (Deepgram).
    Represents one utterance result — partial or final.

    Only final transcripts should be forwarded to the router.
    """
    text: str
    """The recognized speech text."""

    is_final: bool
    """True when Deepgram has committed this transcript. Use only finals for routing."""

    language: str = "en"
    """Detected or assumed language code, e.g. 'en' or 'ar-EG'."""

    confidence: float = 0.0
    """STT confidence score from 0.0 to 1.0."""

    session_id: Optional[str] = None
    """Session identifier this transcript belongs to."""

    source: str = "deepgram"
    """STT provider. Always 'deepgram' in MVP."""


# ─────────────────────────────────────────────────────────────────────────────
# Intent Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntentResult:
    """
    Produced by the Groq router.
    Drives which path the conversation controller takes.
    """
    intent: IntentClass
    """Classified intent for this utterance."""

    language: str = "en"
    """Language detected in the utterance."""

    target_text: Optional[str] = None
    """Extracted entity text for campus or navigation queries, e.g. 'Robotics Lab'."""

    needs_clarification: bool = False
    """True when the router could not confidently pick a single intent or entity."""

    clarification_question: Optional[str] = None
    """The clarification question to ask the user, if needs_clarification is True."""

    confidence: float = 0.0
    """Router confidence score from 0.0 to 1.0."""

    raw_query: Optional[str] = None
    """The original transcript text this result was derived from."""

    reason: Optional[str] = None
    """Optional short reason string from the router, useful for debugging."""


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpokenFacts:
    """
    Structured campus facts safe to pass to the response composer.
    Only verified fields from the database are included here.
    """
    building: Optional[str] = None
    floor: Optional[str] = None
    room: Optional[str] = None
    description: Optional[str] = None
    office_hours: Optional[str] = None
    contact_notes: Optional[str] = None


@dataclass(frozen=True)
class RetrievalResult:
    """
    Produced by the SQLite truth layer.
    The response composer must only use facts from this object — never invent.
    """
    status: RetrievalStatus
    """Outcome of the retrieval attempt."""

    entity_type: Optional[str] = None
    """e.g. 'location', 'staff', 'department', 'facility'"""

    entity_id: Optional[int] = None
    """Primary key of the matched record in SQLite."""

    canonical_name: Optional[str] = None
    """Official name of the matched entity."""

    spoken_facts: Optional[SpokenFacts] = None
    """Structured facts to be used in response generation."""

    nav_code: Optional[str] = None
    """Navigation target code for the action bridge, e.g. 'LAB_214'."""

    map_node: Optional[str] = None
    """Internal map graph node identifier."""

    confidence: float = 0.0
    """Retrieval confidence from 0.0 to 1.0."""

    candidates: list[str] = field(default_factory=list)
    """Candidate names when status is AMBIGUOUS. Used to build clarification question."""

    matched_alias: Optional[str] = None
    """The alias or field that produced this match, useful for debugging."""

    matched_via: Optional[str] = None
    """How the match was found: alias, fts_locations, fts_staff, etc."""


# ─────────────────────────────────────────────────────────────────────────────
# Response Packet
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ResponsePacket:
    """
    The final output of the conversation controller.
    Forwarded to TTS and the action bridge.
    """
    text: str
    """The spoken response text. Must be short and TTS-friendly."""

    language: str = "en"
    """Language code for TTS voice selection."""

    should_navigate: bool = False
    """If True, a navigation command must be emitted alongside TTS playback."""

    navigation_command: Optional["NavigationCommand"] = None
    """The navigation command to emit, if should_navigate is True."""

    session_id: Optional[str] = None
    """Session identifier this response belongs to."""


# ─────────────────────────────────────────────────────────────────────────────
# Navigation Command
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NavigationCommand:
    """
    Sent to the hardware action bridge.
    Must only be constructed from a verified RetrievalResult with a nav_code.
    """
    action: str
    """Command type. MVP supports: 'navigate', 'cancel_navigation'."""

    target_code: str
    """Canonical navigation code from the database, e.g. 'LAB_214'."""

    target_label: str
    """Human-readable destination label for confirmation speech."""

    spoken_confirmation: str
    """The spoken line already sent to TTS confirming this navigation."""

    session_id: Optional[str] = None
    """Session identifier for traceability."""

    safety_mode: str = "standard"
    """Safety mode hint for the hardware team. Always 'standard' in MVP."""


# ─────────────────────────────────────────────────────────────────────────────
# System Error
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemError:
    """
    Structured error event passed between modules.
    Never raise bare exceptions across module boundaries — wrap them here.
    """
    code: str
    """Short machine-readable error code, e.g. 'stt_timeout', 'retrieval_failed'."""

    message: str
    """Human-readable description of the error."""

    source: str
    """Module that generated this error, e.g. 'deepgram_stt', 'groq_router'."""

    recoverable: bool = True
    """If False, the session should be terminated and the system reset."""

    session_id: Optional[str] = None
    """Session identifier if the error occurred inside an active session."""

    detail: Optional[str] = None
    """Optional extra detail for debugging (exception message, raw response, etc.)."""
