"""
Navigator - Phase 9 Tests: Action Bridge and Navigation Bridge

Tests for CommandBus (safety checks, mock emit, payload, ack handling)
and NavigationBridge (spoken error messages, accepted path, no-code block).

No real HTTP calls are made in these tests.

Run with:
    pytest tests/unit/test_actions.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from app.actions.command_bus import AckStatus, CommandBus
from app.actions.navigation_bridge import NavigationBridge
from app.utils.contracts import NavigationCommand


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_cmd(
    action: str = "navigate",
    target_code: str = "NAV_LAB_214",
    target_label: str = "Robotics Lab",
    session_id: str = "s1",
) -> NavigationCommand:
    return NavigationCommand(
        action=action,
        target_code=target_code,
        target_label=target_label,
        spoken_confirmation="Guiding you to the Robotics Lab now.",
        session_id=session_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CommandBus
# ─────────────────────────────────────────────────────────────────────────────

class TestCommandBusMock:
    def test_mock_emit_returns_accepted(self):
        bus = CommandBus(mock=True)
        ack = bus.emit(_make_cmd())
        assert ack == AckStatus.ACCEPTED

    def test_mock_cancel_navigation(self):
        bus = CommandBus(mock=True)
        cmd = NavigationCommand(
            action="cancel_navigation",
            target_code="NAV_LAB_214",
            target_label="Robotics Lab",
            spoken_confirmation="Cancelling navigation.",
        )
        ack = bus.emit(cmd)
        assert ack == AckStatus.ACCEPTED

    def test_empty_target_code_returns_rejected(self):
        bus = CommandBus(mock=True)
        cmd = _make_cmd(target_code="")
        ack = bus.emit(cmd)
        assert ack == AckStatus.REJECTED

    def test_unknown_action_returns_rejected(self):
        bus = CommandBus(mock=True)
        cmd = _make_cmd(action="launch_rockets")
        ack = bus.emit(cmd)
        assert ack == AckStatus.REJECTED

    def test_repeat_last_destination_is_allowed(self):
        bus = CommandBus(mock=True)
        cmd = _make_cmd(action="repeat_last_destination")
        ack = bus.emit(cmd)
        assert ack == AckStatus.ACCEPTED

    def test_payload_built_correctly(self):
        bus = CommandBus(mock=True)
        cmd = _make_cmd()
        payload = bus._build_payload(cmd)
        assert payload["action"]       == "navigate"
        assert payload["target_code"]  == "NAV_LAB_214"
        assert payload["target_label"] == "Robotics Lab"
        assert payload["session_id"]   == "s1"
        assert payload["safety_mode"]  == "standard"

    def test_http_post_accepted_ack(self):
        """Simulate a successful HTTP response returning 'accepted'."""
        bus = CommandBus(mock=False, endpoint="http://localhost:8765/navigate")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "accepted"}
        mock_response.raise_for_status = MagicMock()

        with patch("app.actions.command_bus.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            ack = bus.emit(_make_cmd())

        assert ack == AckStatus.ACCEPTED

    def test_http_post_busy_ack(self):
        bus = CommandBus(mock=False, endpoint="http://localhost:8765/navigate")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "busy"}
        mock_response.raise_for_status = MagicMock()

        with patch("app.actions.command_bus.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            ack = bus.emit(_make_cmd())

        assert ack == AckStatus.BUSY

    def test_http_post_rejected_ack(self):
        bus = CommandBus(mock=False, endpoint="http://localhost:8765/navigate")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "rejected"}
        mock_response.raise_for_status = MagicMock()

        with patch("app.actions.command_bus.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            ack = bus.emit(_make_cmd())

        assert ack == AckStatus.REJECTED

    def test_http_connection_error_returns_error(self):
        bus = CommandBus(mock=False, endpoint="http://localhost:9999/navigate")
        with patch("app.actions.command_bus.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionRefusedError("refused")
            ack = bus.emit(_make_cmd())
        assert ack == AckStatus.ERROR

    def test_unknown_ack_status_returns_error(self):
        bus = CommandBus(mock=False, endpoint="http://localhost:8765/navigate")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "something_unknown"}
        mock_response.raise_for_status = MagicMock()

        with patch("app.actions.command_bus.httpx") as mock_httpx:
            mock_httpx.post.return_value = mock_response
            ack = bus.emit(_make_cmd())

        assert ack == AckStatus.ERROR


# ─────────────────────────────────────────────────────────────────────────────
# NavigationBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestNavigationBridge:
    def test_accepted_returns_none(self):
        bridge = NavigationBridge(mock=True)
        result = bridge.navigate(_make_cmd(), language="en")
        assert result is None

    def test_rejected_returns_english_message(self):
        bridge = NavigationBridge(mock=False)
        bridge._bus = CommandBus(mock=False)
        bridge._bus._mock = False
        with patch.object(bridge._bus, "emit", return_value=AckStatus.REJECTED):
            result = bridge.navigate(_make_cmd(), language="en")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rejected_returns_arabic_message(self):
        bridge = NavigationBridge(mock=False)
        with patch.object(bridge._bus, "emit", return_value=AckStatus.REJECTED):
            result = bridge.navigate(_make_cmd(), language="ar-EG")
        assert isinstance(result, str)
        assert any(ord(c) > 127 for c in result)

    def test_busy_returns_message(self):
        bridge = NavigationBridge(mock=False)
        with patch.object(bridge._bus, "emit", return_value=AckStatus.BUSY):
            result = bridge.navigate(_make_cmd(), language="en")
        assert result is not None
        assert "moment" in result.lower() or "last" in result.lower()

    def test_unknown_target_returns_message(self):
        bridge = NavigationBridge(mock=False)
        with patch.object(bridge._bus, "emit", return_value=AckStatus.UNKNOWN_TARGET):
            result = bridge.navigate(_make_cmd(), language="en")
        assert "navigation" in result.lower() or "path" in result.lower()

    def test_empty_nav_code_blocked(self):
        bridge = NavigationBridge(mock=True)
        cmd = _make_cmd(target_code="")
        result = bridge.navigate(cmd, language="en")
        assert isinstance(result, str)
        assert result is not None

    def test_empty_nav_code_blocked_arabic(self):
        bridge = NavigationBridge(mock=True)
        cmd = _make_cmd(target_code="")
        result = bridge.navigate(cmd, language="ar-EG")
        assert any(ord(c) > 127 for c in result)

    def test_error_ack_returns_error_message(self):
        bridge = NavigationBridge(mock=False)
        with patch.object(bridge._bus, "emit", return_value=AckStatus.ERROR):
            result = bridge.navigate(_make_cmd(), language="en")
        assert isinstance(result, str)
