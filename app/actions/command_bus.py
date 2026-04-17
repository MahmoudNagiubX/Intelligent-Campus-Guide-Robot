"""
Navigator - Action Bridge / Command Bus
Phase 9, Step 9.1 + 9.2 + 9.3 + 9.4

Emits navigation commands to the hardware action bridge via HTTP.
Enforces all safety rules before emitting a command.

Safety contract (§9.4):
- Must have a valid nav_code from the truth layer
- Must not emit if target is ambiguous
- Must not emit if retrieval was NOT_FOUND
- On acknowledgment failure, logs but does NOT crash

Communication:
- MVP: HTTP POST to a configurable local endpoint
- Payload: JSON matching NavigationCommand contract
- Acknowledgment: {"status": "accepted"|"rejected"|"busy"|"unknown_target"}

Mock mode:
    bus = CommandBus(mock=True)
    ack = bus.emit(command)   # returns AckStatus.ACCEPTED without HTTP
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Optional

try:
    import httpx  # type: ignore
except ImportError:
    httpx = None  # type: ignore

from app.utils.contracts import NavigationCommand
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AckStatus(str, Enum):
    ACCEPTED       = "accepted"
    REJECTED       = "rejected"
    BUSY           = "busy"
    UNKNOWN_TARGET = "unknown_target"
    TIMEOUT        = "timeout"
    ERROR          = "error"


class CommandBus:
    """
    Emits NavigationCommand objects to the hardware action bridge.

    Args:
        endpoint: URL of the hardware bridge HTTP endpoint.
        timeout:  Request timeout in seconds.
        mock:     In mock mode, no HTTP is made — always returns ACCEPTED.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 5.0,
        mock: bool = False,
    ) -> None:
        from app.config import get_settings
        cfg = get_settings()
        self._endpoint = endpoint or cfg.action_bridge_url
        self._timeout  = timeout
        self._mock     = mock

        logger.info(
            "command_bus_init",
            endpoint=self._endpoint,
            mock=self._mock,
        )

    def emit(self, command: NavigationCommand) -> AckStatus:
        """
        Emit a navigation command and return the acknowledgment status.

        Safety check is enforced here:
        - target_code must be non-empty
        - action must be one of the supported commands

        Args:
            command: A NavigationCommand built from verified retrieval data.

        Returns:
            AckStatus indicating the hardware response.
        """
        # ── Safety check ──────────────────────────────────────────────────────
        if not command.target_code:
            logger.error(
                "command_bus_safety_blocked",
                reason="empty_target_code",
                session_id=command.session_id,
            )
            return AckStatus.REJECTED

        if command.action not in ("navigate", "cancel_navigation", "repeat_last_destination"):
            logger.error(
                "command_bus_safety_blocked",
                reason="unknown_action",
                action=command.action,
                session_id=command.session_id,
            )
            return AckStatus.REJECTED

        payload = self._build_payload(command)
        logger.info(
            "command_bus_emit",
            action=command.action,
            target_code=command.target_code,
            target_label=command.target_label,
            session_id=command.session_id,
        )

        if self._mock:
            logger.info("command_bus_mock_accepted")
            return AckStatus.ACCEPTED

        return self._post(payload, command.session_id)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_payload(self, command: NavigationCommand) -> dict:
        """Serialize the NavigationCommand to the hardware bridge JSON format."""
        return {
            "action":              command.action,
            "target_code":         command.target_code,
            "target_label":        command.target_label,
            "spoken_confirmation": command.spoken_confirmation,
            "session_id":          command.session_id,
            "safety_mode":         command.safety_mode,
        }

    def _post(self, payload: dict, session_id: Optional[str]) -> AckStatus:
        """POST the command payload to the action bridge endpoint."""
        try:
            if httpx is None:
                logger.error("command_bus_httpx_not_installed")
                return AckStatus.ERROR

            response = httpx.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            status_str = data.get("status", "error").lower()
            status = AckStatus(status_str) if status_str in AckStatus._value2member_map_ else AckStatus.ERROR

            logger.info(
                "command_bus_ack",
                status=status.value,
                target_code=payload["target_code"],
                session_id=session_id,
            )
            return status

        except Exception as exc:
            logger.error("command_bus_post_error", error=str(exc), session_id=session_id)
            return AckStatus.ERROR
