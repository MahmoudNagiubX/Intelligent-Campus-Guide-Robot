from __future__ import annotations

import json

import pytest

from app.pipeline.controller import ConversationController
from app.pipeline.pipecat_graph import NavigatorPipecatRuntime
from app.utils.contracts import SessionState
from tests.fixtures.runtime_fixtures import bootstrap_and_sync, configure_test_settings, simulate_user_turn


class ArabicHybridGroq:
    def complete_json(self, *args, **kwargs) -> str:
        text = kwargs.get("user_message", "")
        if "اتاسست" in text or "اتأسست" in text:
            intent = "Unknown"
            target = ""
        elif "كلية" in text or "هندسة" in text:
            intent = "Campus_Query"
            target = "كلية الهندسة"
        elif "خدني" in text:
            intent = "Navigation_Request"
            target = "معمل الروبوتات"
        else:
            intent = "Campus_Query"
            target = "معمل الروبوتات"
        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.9,
                "language": "ar",
                "needs_retrieval": intent != "Unknown",
                "needs_navigation": intent == "Navigation_Request",
                "target_entity": target,
                "target_type": "none",
                "normalized_query": text,
                "clarification_needed": False,
                "clarification_question": "",
            },
            ensure_ascii=False,
        )

    def complete_text(self, *args, **kwargs) -> str:
        message = kwargs.get("user_message", "")
        if "كلية الهندسة" in message:
            return "كلية الهندسة والتكنولوجيا موجودة ضمن كليات ECU."
        if "اتأسست" in message or "اتاسست" in message:
            return "مش متأكد بالظبط من السنة. جرب تتأكد من ecu.edu.eg."
        return "معمل الروبوتات في Building C، الدور الأول، أوضة C105."


async def _run_turn(monkeypatch, tmp_path, transcript: str) -> list[str]:
    configure_test_settings(monkeypatch, tmp_path, session_timeout=1)
    bootstrap_and_sync()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: ArabicHybridGroq())
    runtime = NavigatorPipecatRuntime(
        mock=True,
        auto_start_audio=False,
        controller=ConversationController(groq=ArabicHybridGroq()),
    )
    await runtime.start()
    try:
        await simulate_user_turn(runtime, transcript, language="ar-EG")
        assert await runtime.wait_for_state(SessionState.IDLE, timeout=3.0)
        return [event.data.get("text", "") for event in runtime.tracer.events() if event.name == "response_generated"]
    finally:
        await runtime.shutdown()


def _assert_no_raw_not_found(responses: list[str]) -> None:
    joined = " ".join(responses)
    assert "مش لاقي" not in joined
    assert "غير موجود في قاعدة البيانات" not in joined


@pytest.mark.asyncio
async def test_direct_arabic_db_match(monkeypatch, tmp_path) -> None:
    responses = await _run_turn(monkeypatch, tmp_path, "فين معمل الروبوتات")

    assert any("أوضة" in text or "Building" in text for text in responses)
    _assert_no_raw_not_found(responses)


@pytest.mark.asyncio
async def test_semantic_arabic_db_match(monkeypatch, tmp_path) -> None:
    responses = await _run_turn(monkeypatch, tmp_path, "عايز أعرف عن معمل الروبوتيكس")

    assert any("أوضة" in text or "Building" in text for text in responses)
    _assert_no_raw_not_found(responses)


@pytest.mark.asyncio
async def test_arabic_ecu_cache_fallback(monkeypatch, tmp_path) -> None:
    responses = await _run_turn(monkeypatch, tmp_path, "الجامعة فيها كلية هندسة؟")

    assert any("الهندسة" in text for text in responses)
    _assert_no_raw_not_found(responses)


@pytest.mark.asyncio
async def test_arabic_general_fallback(monkeypatch, tmp_path) -> None:
    responses = await _run_turn(monkeypatch, tmp_path, "الجامعة اتأسست امتى؟")

    assert any(any("\u0600" <= ch <= "\u06FF" for ch in text) for text in responses)
    assert any("ecu.edu.eg" in text for text in responses)
    _assert_no_raw_not_found(responses)
