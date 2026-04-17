"""
Navigator - Navigation Bridge (High-Level Action Interface)
Phase 9

High-level interface used by the conversation controller to trigger navigation.
Wraps CommandBus with additional safety logic and spoken confirmation.

Safety rules enforced here:
1. nav_code must be present and non-empty
2. ambiguous retrievals must be resolved before calling this
3. if CommandBus returns BUSY or REJECTED, provide user feedback

Usage:
    bridge = NavigationBridge()
    response_text = bridge.navigate(nav_command, session_id)
"""

from __future__ import annotations

from typing import Optional

from app.actions.command_bus import AckStatus, CommandBus
from app.utils.contracts import NavigationCommand
from app.utils.logging import get_logger

logger = get_logger(__name__)

_ACK_MESSAGES_EN: dict[AckStatus, str] = {
    AckStatus.ACCEPTED:       "",   # spoken_confirmation already emitted in TTS
    AckStatus.REJECTED:       "Sorry, I couldn't start navigation right now. Please try again.",
    AckStatus.BUSY:           "I'm still finishing the last route. Give me a moment.",
    AckStatus.UNKNOWN_TARGET: "I don't have a path to that location in my navigation system.",
    AckStatus.TIMEOUT:        "Navigation didn't respond in time. Please try again.",
    AckStatus.ERROR:          "Something went wrong with navigation. Let's try again.",
}

_ACK_MESSAGES_AR: dict[AckStatus, str] = {
    AckStatus.ACCEPTED:       "",
    AckStatus.REJECTED:       "معلش، مقدرتش أبدأ التوجيه دلوقتي. جرب تاني.",
    AckStatus.BUSY:           "لسه بخلص المسار اللي قبل. لحظة.",
    AckStatus.UNKNOWN_TARGET:  "مش عندي مسار لهذا المكان في نظام التوجيه.",
    AckStatus.TIMEOUT:        "النظام مجاوبش. جرب تاني.",
    AckStatus.ERROR:          "حصل خطأ في التوجيه. نجرب تاني.",
}


class NavigationBridge:
    """
    High-level navigation interface for the conversation controller.

    Args:
        mock: If True, the underlying CommandBus is also in mock mode.
    """

    def __init__(self, mock: bool = False) -> None:
        self._bus  = CommandBus(mock=mock)
        self._mock = mock
        logger.info("navigation_bridge_init", mock=mock)

    def navigate(
        self,
        command: NavigationCommand,
        language: str = "en",
    ) -> Optional[str]:
        """
        Execute a navigation command and return a supplementary spoken message
        if the action bridge returned anything other than ACCEPTED.

        For ACCEPTED: returns None (spoken_confirmation already played via TTS).
        For errors: returns a localized spoken fallback.

        Safety: target_code must be non-empty (enforced in CommandBus).

        Args:
            command:  Validated NavigationCommand from the composer.
            language: Language for error messages.

        Returns:
            None if navigation was accepted, or supplementary spoken text on failure.
        """
        if not command.target_code:
            logger.error(
                "nav_bridge_blocked_no_nav_code",
                label=command.target_label,
                session_id=command.session_id,
            )
            msgs = _ACK_MESSAGES_AR if language == "ar-EG" else _ACK_MESSAGES_EN
            return msgs[AckStatus.UNKNOWN_TARGET]

        ack = self._bus.emit(command)

        logger.info(
            "nav_bridge_result",
            ack=ack.value,
            target_code=command.target_code,
            language=language,
            session_id=command.session_id,
        )

        if ack == AckStatus.ACCEPTED:
            return None  # TTS already spoke the confirmation

        msgs = _ACK_MESSAGES_AR if language == "ar-EG" else _ACK_MESSAGES_EN
        return msgs.get(ack, msgs[AckStatus.ERROR])
