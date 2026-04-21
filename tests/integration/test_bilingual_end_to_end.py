from __future__ import annotations

import json

import pytest

from app.actions.navigation_bridge import NavigationBridge
from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.stt.dual_stt_client import DualSTTClient
from app.storage.db import get_db
from app.utils.contracts import SessionState, TranscriptEvent
from tests.fixtures.runtime_fixtures import bootstrap_and_sync, configure_test_settings, simulate_user_turn


class CampusGroq:
    def complete_text(self, *args, **kwargs) -> str:
        prompt = kwargs.get("system_prompt", "")
        return "المعمل في المبنى C، أوضة 214." if any("\u0600" <= ch <= "\u06FF" for ch in prompt) else "The lab is in Building C, room 214."


class RouterGroq:
    def complete_json(self, *args, **kwargs) -> str:
        text = kwargs.get("user_message", "")
        is_ar = any("\u0600" <= ch <= "\u06FF" for ch in text)
        nav = "خدني" in text or "take me" in text
        return json.dumps(
            {
                "intent": "Navigation_Request" if nav else "Campus_Query",
                "confidence": 0.92,
                "language": "ar" if is_ar else "en",
                "needs_retrieval": True,
                "needs_navigation": nav,
                "target_entity": "Robotics Lab" if not is_ar else "معمل الروبوتات",
                "target_type": "lab",
                "normalized_query": text,
                "clarification_needed": False,
                "clarification_question": "",
            },
            ensure_ascii=False,
        )


@pytest.mark.asyncio
async def test_english_campus_query(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: RouterGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=CampusGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "where is the robotics lab", language="en")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any("Building" in text or "building" in text for text in responses)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_arabic_campus_query(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: RouterGroq())

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=CampusGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "فين معمل الروبوتات", language="ar-EG")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        responses = [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
        assert any(any("\u0600" <= ch <= "\u06FF" for ch in text) for text in responses)
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_arabic_navigation_request(monkeypatch, tmp_path):
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    conn = get_db()
    labs = conn.execute("SELECT id FROM labs").fetchall()
    for lab in labs:
        conn.execute(
            "INSERT OR IGNORE INTO navigation_targets (target_type, canonical_id, nav_code, updated_at) VALUES ('lab', ?, ?, datetime('now'))",
            (lab["id"], f"NAV_LAB_{lab['id']}"),
        )
    rooms = conn.execute("SELECT id FROM rooms").fetchall()
    for room in rooms:
        conn.execute(
            "INSERT OR IGNORE INTO navigation_targets (target_type, canonical_id, nav_code, updated_at) VALUES ('room', ?, ?, datetime('now'))",
            (room["id"], f"NAV_ROOM_{room['id']}"),
        )
    conn.commit()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: RouterGroq())

    captured = {}

    class RecordingBridge(NavigationBridge):
        def navigate(self, command, language="en"):
            captured["target_code"] = command.target_code
            return None

    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=CampusGroq()),
        navigation_bridge=RecordingBridge(mock=True),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, "خدني لمعمل الروبوتيكس", language="ar-EG")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        assert captured.get("target_code")
    finally:
        await runtime.shutdown()


def test_phonetic_arabic_resolved(monkeypatch):
    monkeypatch.setattr("app.stt.dual_stt_client.load_arabic_keyterms_from_db", lambda: [])
    finals = []
    client = DualSTTClient(mock=True, on_final=finals.append, session_id="sess")
    client._on_deepgram_final(
        TranscriptEvent(
            text="wayn almaktab",
            is_final=True,
            language="en",
            language_confidence=0.9,
            confidence=0.95,
            session_id="sess",
            source="deepgram",
        )
    )
    client._on_arabic_final(
        TranscriptEvent(
            text="وين المكتب",
            is_final=True,
            language="ar-EG",
            language_confidence=0.95,
            confidence=0.95,
            session_id="sess",
            source="elevenlabs",
        )
    )

    assert len(finals) == 1
    assert finals[0].language == "ar-EG"
    assert finals[0].text == "وين المكتب"
