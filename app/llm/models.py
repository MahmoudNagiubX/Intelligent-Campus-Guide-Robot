"""
Navigator - Groq Router Response Models

Pydantic models for validating and normalizing the raw JSON object returned
by the Groq LLM router.  This module is the only place that touches raw model
output — everything downstream works with typed IntentResult objects.

Rules:
- This module must not import from app.llm (no circular dependency).
- Validation always normalizes before rejecting — uppercase intent strings are
  accepted and lowercased rather than rejected outright.
- Invalid output raises pydantic.ValidationError; callers must handle it.
"""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.utils.contracts import IntentClass, IntentResult


class RouterRawOutput(BaseModel):
    """
    Validates the raw JSON object returned by the Groq router completion.

    The model normalizes case-insensitive intent values and clamps confidence
    to [0.0, 1.0] before any downstream code sees the data.
    """

    intent: str
    """One of the four locked intent classes (case-insensitive on input)."""

    language: str = "en"
    """Language code detected in the utterance, e.g. 'en' or 'ar-EG'."""

    target_text: Optional[str] = None
    """Extracted entity name for campus/navigation queries, e.g. 'Robotics Lab'."""

    needs_clarification: bool = False
    """True when the router could not confidently resolve intent or entity."""

    clarification_question: Optional[str] = None
    """Clarification question to surface to the user, when needs_clarification=True."""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """Router classification confidence score, clamped to [0.0, 1.0]."""

    reason: Optional[str] = None
    """Short debug string explaining the classification decision."""

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, v: object) -> str:
        """Normalize and validate the intent string against locked intent classes."""
        normalized = str(v).lower().strip()
        valid = {ic.value for ic in IntentClass}
        if normalized not in valid:
            raise ValueError(
                f"Unknown intent {v!r}. Valid values: {sorted(valid)}"
            )
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        """Clamp confidence to [0.0, 1.0]; return 0.0 for non-numeric input."""
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    def to_intent_result(self, raw_query: Optional[str] = None) -> IntentResult:
        """Convert this validated model to the shared IntentResult contract."""
        return IntentResult(
            intent=IntentClass(self.intent),
            language=self.language,
            target_text=self.target_text or None,
            needs_clarification=self.needs_clarification,
            clarification_question=self.clarification_question or None,
            confidence=self.confidence,
            raw_query=raw_query,
            reason=self.reason,
        )


def parse_router_response(
    raw_json: str,
    raw_query: Optional[str] = None,
) -> IntentResult | None:
    """
    Parse and validate a raw JSON string from Groq into an IntentResult.

    Returns None if:
    - the string is not valid JSON
    - the JSON contains an unknown intent value
    - any required field is missing or malformed

    Callers are expected to substitute a safe UNKNOWN IntentResult when None
    is returned.
    """
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    try:
        validated = RouterRawOutput.model_validate(data)
        return validated.to_intent_result(raw_query=raw_query)
    except Exception:
        return None
