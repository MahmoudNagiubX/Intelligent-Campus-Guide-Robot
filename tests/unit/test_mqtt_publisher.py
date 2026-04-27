"""
Unit tests for MQTTPublisher.

Tests:
- State mapping (idle/error → "wait", wake/listening/processing → "listen", speaking → "speak")
- Duplicate states are not published twice
- Unknown states are silently ignored
- Disabled publisher never connects or publishes
- publish_state() and stop() never raise exceptions even when paho is missing
- publish_state() skips when client is None (broker unreachable)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from app.ui.mqtt_publisher import MQTTPublisher, _STATE_MAP


# ------------------------------------------------------------------
# State mapping
# ------------------------------------------------------------------


def test_state_map_idle_to_wait():
    assert _STATE_MAP["idle"] == "wait"


def test_state_map_error_to_wait():
    assert _STATE_MAP["error"] == "wait"


def test_state_map_wake_detected_to_listen():
    assert _STATE_MAP["wake_detected"] == "listen"


def test_state_map_listening_to_listen():
    assert _STATE_MAP["listening"] == "listen"


def test_state_map_processing_to_listen():
    assert _STATE_MAP["processing"] == "listen"


def test_state_map_speaking_to_speak():
    assert _STATE_MAP["speaking"] == "speak"


# ------------------------------------------------------------------
# Disabled publisher — no connection, no-op
# ------------------------------------------------------------------


def test_disabled_publisher_does_not_connect():
    pub = MQTTPublisher(enabled=False, broker="mqtt.example.com")
    pub.start()
    assert pub._client is None, "Disabled publisher must never create a client"


def test_disabled_publisher_publish_state_is_noop():
    pub = MQTTPublisher(enabled=False, broker="mqtt.example.com")
    pub.publish_state("idle")  # must not raise


def test_disabled_publisher_stop_is_noop():
    pub = MQTTPublisher(enabled=False, broker="mqtt.example.com")
    pub.stop()  # must not raise


def test_enabled_without_broker_does_not_connect():
    pub = MQTTPublisher(enabled=True, broker="")
    pub.start()
    assert pub._client is None, "Empty broker must never create a client"


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------


def test_duplicate_state_not_published_twice():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish_state("idle")
    pub.publish_state("idle")

    mock_client.publish.assert_called_once()


def test_different_states_both_published():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish_state("idle")
    pub.publish_state("speaking")

    assert mock_client.publish.call_count == 2


def test_same_mqtt_string_from_different_source_states_deduped():
    """wake_detected and listening both map to "listen" — second should be skipped."""
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish_state("wake_detected")
    pub.publish_state("listening")

    mock_client.publish.assert_called_once()


def test_state_sequence_publishes_correctly():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish_state("idle")       # → "wait"
    pub.publish_state("listening")  # → "listen"
    pub.publish_state("speaking")   # → "speak"
    pub.publish_state("idle")       # → "wait"

    assert mock_client.publish.call_count == 4
    calls = [c.args[1] for c in mock_client.publish.call_args_list]
    assert calls == ["wait", "listen", "speak", "wait"]


# ------------------------------------------------------------------
# Unknown states
# ------------------------------------------------------------------


def test_unknown_state_silently_ignored():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    pub._client = mock_client

    pub.publish_state("nonexistent_state")  # must not raise, must not publish
    mock_client.publish.assert_not_called()


# ------------------------------------------------------------------
# No client (broker unreachable)
# ------------------------------------------------------------------


def test_publish_with_no_client_does_not_raise():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    # _client is None (connect_safe failed silently)
    pub.publish_state("speaking")  # must not raise


def test_stop_with_no_client_does_not_raise():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    pub.stop()  # must not raise


# ------------------------------------------------------------------
# paho-mqtt not installed
# ------------------------------------------------------------------


def test_start_without_paho_logs_warning_and_continues():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    import sys

    fake_modules = {k: None for k in list(sys.modules) if k.startswith("paho")}
    # Remove paho from sys.modules so the import inside _connect_safe fails
    with patch.dict("sys.modules", {"paho": None, "paho.mqtt": None, "paho.mqtt.client": None}):
        try:
            pub._connect_safe()
        except Exception as exc:
            pytest.fail(f"_connect_safe raised unexpectedly: {exc}")
    assert pub._client is None


# ------------------------------------------------------------------
# stop() publishes "wait"
# ------------------------------------------------------------------


def test_stop_publishes_wait_and_disconnects():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com", topic="test/topic", qos=1, retain=True)
    mock_client = MagicMock()
    pub._client = mock_client

    pub.stop()

    mock_client.publish.assert_called_once_with("test/topic", "wait", qos=1, retain=True)
    mock_client.loop_stop.assert_called_once()
    mock_client.disconnect.assert_called_once()


def test_stop_does_not_raise_when_client_raises():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    mock_client.publish.side_effect = RuntimeError("network gone")
    pub._client = mock_client

    pub.stop()  # must not propagate


# ------------------------------------------------------------------
# publish errors do not raise
# ------------------------------------------------------------------


def test_publish_error_does_not_raise():
    pub = MQTTPublisher(enabled=True, broker="mqtt.example.com")
    mock_client = MagicMock()
    mock_client.publish.side_effect = RuntimeError("publish failed")
    pub._client = mock_client

    pub.publish_state("idle")  # must not raise
