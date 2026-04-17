"""
Navigator - Groq LLM Client

Wrapper around the Groq Python SDK. All tunable parameters come from config.

Provides three public methods:
- complete_json(): sends a prompt and returns a raw JSON string
- complete_text(): sends a prompt and returns plain text
- call_router(): classifies intent and always returns IntentResult

Rules:
- JSON mode is used only when the caller expects structured output
- call_router() never raises
- Retries use exponential backoff
- Pipeline code should use higher-level services rather than calling the SDK
  directly
"""

from __future__ import annotations

import json
import time
from typing import Optional

from groq import APIConnectionError, APITimeoutError, Groq, RateLimitError
from pydantic import ValidationError

from app.config import get_settings
from app.llm.models import RouterRawOutput
from app.utils.contracts import IntentClass, IntentResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


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


class GroqClient:
    """
    Wrapper around the Groq API client.

    Instantiate once at startup and reuse across all turns in the session.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        if not cfg.has_groq_key:
            raise RuntimeError("GROQ_API_KEY is not set - cannot initialize Groq client.")

        self._model: str = cfg.groq_model
        self._timeout: float = cfg.groq_timeout
        self._max_retries: int = cfg.groq_max_retries
        self._retry_backoff: float = cfg.groq_retry_backoff

        self._client = Groq(api_key=cfg.groq_api_key, timeout=self._timeout)
        logger.info("groq_client_ready", model=self._model, timeout=self._timeout)

    def close(self) -> None:
        """Close the underlying Groq/httpx client explicitly."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        self._client = None
        client.close()

    def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> str | None:
        """Send a completion request in JSON mode."""
        return self._complete_request(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            mode="json",
        )

    def complete_text(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 256,
    ) -> str | None:
        """Send a completion request in plain-text mode."""
        return self._complete_request(
            system_prompt=system_prompt,
            user_message=user_message,
            max_tokens=max_tokens,
            temperature=0.2,
            response_format=None,
            mode="text",
        )

    def _complete_request(
        self,
        *,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
        temperature: float,
        response_format: Optional[dict],
        mode: str,
    ) -> str | None:
        """Shared completion logic for JSON and plain-text requests."""
        for attempt in range(1, self._max_retries + 1):
            try:
                request: dict = {
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if response_format is not None:
                    request["response_format"] = response_format

                response = self._client.chat.completions.create(**request)
                content = response.choices[0].message.content
                logger.debug(
                    "groq_response_received",
                    mode=mode,
                    attempt=attempt,
                    tokens_used=response.usage.total_tokens if response.usage else 0,
                )
                return content

            except APITimeoutError:
                logger.warning("groq_timeout", mode=mode, attempt=attempt, max=self._max_retries)

            except RateLimitError:
                logger.warning("groq_rate_limit", mode=mode, attempt=attempt)
                time.sleep(2.0)

            except APIConnectionError as exc:
                logger.error("groq_connection_error", mode=mode, error=str(exc), attempt=attempt)

            except Exception as exc:
                logger.error("groq_unexpected_error", mode=mode, error=str(exc), attempt=attempt)
                break

            if attempt < self._max_retries:
                delay = self._retry_backoff * (2 ** (attempt - 1))
                logger.debug(
                    "groq_retry_wait",
                    mode=mode,
                    delay_sec=round(delay, 2),
                    next_attempt=attempt + 1,
                )
                time.sleep(delay)

        logger.error("groq_all_retries_failed", mode=mode, max=self._max_retries)
        return None

    def call_router(
        self,
        system_prompt: str,
        user_message: str,
    ) -> IntentResult:
        """
        Classify the intent of a user utterance.

        Calls complete_json(), then parses and validates the JSON response
        against RouterRawOutput. Returns UNKNOWN on every failure path.
        """
        raw_json = self.complete_json(system_prompt, user_message)

        if raw_json is None:
            logger.error("groq_router_no_response", transcript_preview=user_message[:80])
            return _unknown_result(raw_query=user_message, reason="no_response")

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error(
                "groq_router_json_parse_error",
                error=str(exc),
                raw_preview=raw_json[:120],
            )
            return _unknown_result(raw_query=user_message, reason="json_parse_error")

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
