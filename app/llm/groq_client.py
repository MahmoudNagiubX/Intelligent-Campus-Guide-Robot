"""
Navigator - Groq LLM Client

Wrapper around the Groq Python SDK.  All tunable parameters come from config;
nothing is hardcoded here.

Provides two public methods:
- complete_json()  — low-level: sends a prompt and returns the raw JSON string.
- call_router()    — high-level: classifies intent and always returns IntentResult.

Rules:
- JSON mode is always enabled; the model must return a JSON object.
- call_router() never raises — it returns UNKNOWN on every failure path.
- Retries use exponential backoff.  Non-retriable errors stop immediately.
- Never called directly from pipeline code — use the routing service (Step 2.4).
"""

import json
import time

from groq import Groq, APITimeoutError, APIConnectionError, RateLimitError
from pydantic import ValidationError

from app.config import get_settings
from app.llm.models import RouterRawOutput
from app.utils.contracts import IntentClass, IntentResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Safe fallback helper
# ─────────────────────────────────────────────────────────────────────────────

def _unknown_result(
    raw_query: str | None = None,
    reason: str = "groq_failure",
) -> IntentResult:
    """Return a safe UNKNOWN IntentResult for any failure path."""
    return IntentResult(
        intent=IntentClass.UNKNOWN,
        language="en",
        raw_query=raw_query,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GroqClient
# ─────────────────────────────────────────────────────────────────────────────

class GroqClient:
    """
    Wrapper around the Groq API client.

    Instantiate once at startup and reuse across all turns in the session.
    All configuration (model, timeout, retries, backoff) comes from Settings.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        if not cfg.has_groq_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set — cannot initialize Groq client."
            )

        self._model: str = cfg.groq_model
        self._timeout: float = cfg.groq_timeout
        self._max_retries: int = cfg.groq_max_retries
        self._retry_backoff: float = cfg.groq_retry_backoff

        self._client = Groq(api_key=cfg.groq_api_key, timeout=self._timeout)
        logger.info("groq_client_ready", model=self._model, timeout=self._timeout)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """
        Explicitly close the underlying Groq/httpx client.
        Prevents SDK destructor noise during interpreter shutdown.
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        self._client = None
        client.close()

    # ── Low-level API ─────────────────────────────────────────────────────────

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> str | None:
        """
        Send a chat completion request in JSON mode.

        Retries up to _max_retries times with exponential backoff.
        Non-retriable errors (unexpected exceptions) stop immediately.

        Args:
            system_prompt: System instruction for the model.
            user_message:  The user utterance or transcript text.
            max_tokens:    Upper bound on response tokens.

        Returns:
            Raw JSON string from the model, or None on all-retry failure.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=max_tokens,
                    temperature=0.0,  # deterministic classification
                )
                content = response.choices[0].message.content
                logger.debug(
                    "groq_response_received",
                    attempt=attempt,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                )
                return content

            except APITimeoutError:
                logger.warning("groq_timeout", attempt=attempt, max=self._max_retries)

            except RateLimitError:
                logger.warning("groq_rate_limit", attempt=attempt)
                time.sleep(2.0)  # fixed back-off for rate limits

            except APIConnectionError as exc:
                logger.error("groq_connection_error", error=str(exc), attempt=attempt)

            except Exception as exc:
                # Unexpected errors are not retried — stop immediately.
                logger.error("groq_unexpected_error", error=str(exc), attempt=attempt)
                break

            if attempt < self._max_retries:
                delay = self._retry_backoff * (2 ** (attempt - 1))
                logger.debug(
                    "groq_retry_wait",
                    delay_sec=round(delay, 2),
                    next_attempt=attempt + 1,
                )
                time.sleep(delay)

        logger.error("groq_all_retries_failed", max=self._max_retries)
        return None

    # ── High-level router interface ───────────────────────────────────────────

    def call_router(
        self,
        system_prompt: str,
        user_message: str,
    ) -> IntentResult:
        """
        Classify the intent of a user utterance.

        Calls complete_json(), then parses and validates the JSON response
        against RouterRawOutput.  Always returns a typed IntentResult —
        never raises.  Returns UNKNOWN on every failure path.

        Args:
            system_prompt: Router classification prompt (provided by routing service).
            user_message:  Final transcript text from the STT module.

        Returns:
            IntentResult with a valid intent class.  Callers must not assume
            the result is anything other than UNKNOWN without checking intent.
        """
        raw_json = self.complete_json(system_prompt, user_message)

        if raw_json is None:
            logger.error(
                "groq_router_no_response",
                transcript_preview=user_message[:80],
            )
            return _unknown_result(raw_query=user_message, reason="no_response")

        # ── JSON parse ────────────────────────────────────────────────────────
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error(
                "groq_router_json_parse_error",
                error=str(exc),
                raw_preview=raw_json[:120],
            )
            return _unknown_result(raw_query=user_message, reason="json_parse_error")

        # ── Schema validation ─────────────────────────────────────────────────
        try:
            validated = RouterRawOutput.model_validate(data)
        except ValidationError as exc:
            logger.error(
                "groq_router_validation_error",
                error=str(exc),
                raw_preview=raw_json[:120],
            )
            return _unknown_result(raw_query=user_message, reason="validation_error")

        result = validated.to_intent_result(raw_query=user_message)
        logger.info(
            "groq_router_classified",
            intent=result.intent.value,
            language=result.language,
            target=result.target_text,
            confidence=result.confidence,
            needs_clarification=result.needs_clarification,
        )
        return result
