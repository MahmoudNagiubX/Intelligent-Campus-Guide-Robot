"""
Unit tests for StatusPublisher.

Tests JSON file creation, atomic writes, correct field values,
and graceful no-op when WebSocket is disabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ui.status_publisher import StatusPublisher


def _publisher(tmp_path: Path, ws_enabled: bool = False) -> StatusPublisher:
    pub = StatusPublisher(
        json_path=str(tmp_path / "status.json"),
        ws_enabled=ws_enabled,
    )
    pub.start()
    return pub


# ------------------------------------------------------------------
# Basic JSON writes
# ------------------------------------------------------------------


def test_publish_creates_json_file(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="idle", state="idle", message="Waiting for wake word")

    path = tmp_path / "status.json"
    assert path.exists(), "JSON file must be created after first publish"


def test_publish_idle_fields(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(
        event="idle",
        state="idle",
        message="Waiting for wake word",
        is_listening=False,
        is_speaking=False,
        wake_word_detected=False,
    )
    data = json.loads((tmp_path / "status.json").read_text())

    assert data["event"] == "idle"
    assert data["state"] == "idle"
    assert data["is_listening"] is False
    assert data["is_speaking"] is False
    assert data["wake_word_detected"] is False
    assert data["session_id"] is None
    assert "timestamp" in data


def test_publish_listening_fields(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(
        event="listening",
        state="listening",
        message="Listening...",
        session_id="test-session-123",
        is_listening=True,
        is_speaking=False,
        wake_word_detected=True,
    )
    data = json.loads((tmp_path / "status.json").read_text())

    assert data["event"] == "listening"
    assert data["is_listening"] is True
    assert data["is_speaking"] is False
    assert data["wake_word_detected"] is True
    assert data["session_id"] == "test-session-123"


def test_publish_speaking_fields(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(
        event="speaking",
        state="speaking",
        message="Speaking...",
        session_id="s1",
    )
    data = json.loads((tmp_path / "status.json").read_text())

    assert data["is_listening"] is False
    assert data["is_speaking"] is True
    assert data["wake_word_detected"] is True


def test_publish_updates_json_on_second_call(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="idle", state="idle", message="Waiting for wake word")
    pub.publish(
        event="listening",
        state="listening",
        message="Listening...",
        session_id="s2",
    )
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["event"] == "listening"
    assert data["session_id"] == "s2"


# ------------------------------------------------------------------
# Auto-derived booleans (when not explicitly passed)
# ------------------------------------------------------------------


def test_auto_derive_is_listening(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="listening", state="listening", message="x")
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["is_listening"] is True
    assert data["is_speaking"] is False


def test_auto_derive_is_speaking(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="speaking", state="speaking", message="x")
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["is_speaking"] is True
    assert data["is_listening"] is False


def test_auto_derive_wake_word_false_for_idle(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="idle", state="idle", message="x")
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["wake_word_detected"] is False


def test_auto_derive_wake_word_true_for_non_idle(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="processing", state="processing", message="x")
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["wake_word_detected"] is True


# ------------------------------------------------------------------
# Extra kwargs are written to the payload
# ------------------------------------------------------------------


def test_extra_kwargs_included(tmp_path):
    pub = _publisher(tmp_path)
    pub.publish(event="idle", state="idle", message="x", custom_field="hello")
    data = json.loads((tmp_path / "status.json").read_text())
    assert data["custom_field"] == "hello"


# ------------------------------------------------------------------
# Fail-safe: bad path should not raise
# ------------------------------------------------------------------


def test_bad_path_does_not_raise():
    pub = StatusPublisher(json_path="/nonexistent_root/\x00/status.json", ws_enabled=False)
    pub.start()
    # publish must never raise even with an unwritable path
    pub.publish(event="idle", state="idle", message="x")


# ------------------------------------------------------------------
# No WebSocket = no error
# ------------------------------------------------------------------


def test_ws_disabled_no_error(tmp_path):
    pub = _publisher(tmp_path, ws_enabled=False)
    pub.publish(event="idle", state="idle", message="No WebSocket")
    pub.stop()  # Should not raise even with ws_enabled=False
