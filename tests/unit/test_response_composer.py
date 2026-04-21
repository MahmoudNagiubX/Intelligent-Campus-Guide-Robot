from __future__ import annotations

from app.pipeline.response_composer import ResponseComposer
from app.utils.contracts import RetrievalResult, RetrievalStatus, SpokenFacts


class RecordingGroq:
    def __init__(self, response_text: str = "stubbed") -> None:
        self.response_text = response_text
        self.text_calls: list[dict] = []

    def complete_text(self, *args, **kwargs) -> str:
        self.text_calls.append({"args": args, "kwargs": kwargs})
        return self.response_text


def _retrieval_ok(*, nav_code: str | None = None, name: str = "Robotics Lab") -> RetrievalResult:
    return RetrievalResult(
        status=RetrievalStatus.OK,
        entity_type="room",
        entity_id=1,
        canonical_name=name,
        spoken_facts=SpokenFacts(building="C", floor="1", room="C105", description="Lab"),
        nav_code=nav_code,
        confidence=0.95,
        matched_via="alias",
    )


def test_compose_campus_answer_calls_groq_for_english() -> None:
    groq = RecordingGroq("The robotics lab is in building C. I can guide you there if you'd like.")
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_campus_answer(
        retrieval=_retrieval_ok(nav_code="NAV_C105"),
        original_query="where is the robotics lab",
        language="en",
        session_id="sess-1",
    )

    assert len(groq.text_calls) == 1
    call = groq.text_calls[0]["kwargs"]
    assert packet.language == "en"
    assert packet.text == "The robotics lab is in building C. I can guide you there if you'd like."
    assert "Name: Robotics Lab" in call["system_prompt"]
    assert "Building: C" in call["system_prompt"]
    assert "Floor: 1" in call["system_prompt"]
    assert "Room: C105" in call["system_prompt"]
    assert "Type: Room" in call["system_prompt"]
    assert "Description: Lab" in call["system_prompt"]
    assert "Nav code: NAV_C105" in call["system_prompt"]
    assert call["user_message"] in call["system_prompt"]


def test_compose_campus_answer_calls_groq_for_arabic() -> None:
    groq = RecordingGroq("معمل الروبوتات في المبنى C والدور 1. أقدر آخدك هناك لو حابب.")
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_campus_answer(
        retrieval=RetrievalResult(
            status=RetrievalStatus.OK,
            entity_type="lab",
            entity_id=2,
            canonical_name="معمل الروبوتات",
            spoken_facts=SpokenFacts(building="C", floor="1", room="C105", description="Lab"),
            nav_code="NAV_C105",
            confidence=0.95,
        ),
        original_query="فين معمل الروبوتات",
        language="ar-EG",
        session_id="sess-2",
    )

    assert len(groq.text_calls) == 1
    call = groq.text_calls[0]["kwargs"]
    assert packet.language == "ar-EG"
    assert packet.text == "معمل الروبوتات في المبنى C والدور 1. أقدر آخدك هناك لو حابب."
    assert "الاسم: معمل الروبوتات" in call["system_prompt"]
    assert "المبنى: C" in call["system_prompt"]
    assert "الدور: 1" in call["system_prompt"]
    assert "الغرفة: C105" in call["system_prompt"]
    assert "النوع: معمل" in call["system_prompt"]
    assert "الوصف: معمل" in call["system_prompt"]
    assert "رمز التوجيه: NAV_C105" in call["system_prompt"]
    assert call["user_message"] in call["system_prompt"]


def test_compose_campus_answer_falls_back_when_groq_returns_empty() -> None:
    groq = RecordingGroq("")
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_campus_answer(
        retrieval=_retrieval_ok(nav_code="NAV_C105"),
        original_query="where is the robotics lab",
        language="en",
        session_id="sess-3",
    )

    assert len(groq.text_calls) == 1
    assert "Robotics Lab" in packet.text
    assert "guide you there" in packet.text


def test_compose_social_answer_still_uses_groq() -> None:
    groq = RecordingGroq("Hey there!")
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_social_answer("hello there", language="en", session_id="sess-4")

    assert len(groq.text_calls) == 1
    assert packet.text == "Hey there!"
