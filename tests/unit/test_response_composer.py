from __future__ import annotations

from app.pipeline.response_composer import ResponseComposer
from app.utils.contracts import RetrievalResult, RetrievalStatus, SpokenFacts


class RecordingGroq:
    def __init__(self) -> None:
        self.text_calls = []

    def complete_text(self, *args, **kwargs) -> str:
        self.text_calls.append({"args": args, "kwargs": kwargs})
        return "stubbed"


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


def test_compose_campus_answer_does_not_call_groq_for_english() -> None:
    groq = RecordingGroq()
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_campus_answer(
        retrieval=_retrieval_ok(nav_code="NAV_C105"),
        original_query="where is the robotics lab",
        language="en",
        session_id="sess-1",
    )

    assert groq.text_calls == []
    assert packet.language == "en"
    assert "Robotics Lab" in packet.text
    assert "building C" in packet.text
    assert "guide you there" in packet.text


def test_compose_campus_answer_does_not_call_groq_for_arabic() -> None:
    groq = RecordingGroq()
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_campus_answer(
        retrieval=RetrievalResult(
            status=RetrievalStatus.OK,
            entity_type="room",
            entity_id=2,
            canonical_name="معمل الروبوتات",
            spoken_facts=SpokenFacts(building="C", floor="1", room="C105"),
            nav_code="NAV_C105",
            confidence=0.95,
        ),
        original_query="فين معمل الروبوتات",
        language="ar-EG",
        session_id="sess-2",
    )

    assert groq.text_calls == []
    assert packet.language == "ar-EG"
    assert "معمل الروبوتات" in packet.text
    assert "المبنى C" in packet.text
    assert "أوصلك" in packet.text


def test_compose_social_answer_still_uses_groq() -> None:
    groq = RecordingGroq()
    composer = ResponseComposer(groq=groq)

    packet = composer.compose_social_answer("hello there", language="en", session_id="sess-3")

    assert len(groq.text_calls) == 1
    assert packet.text == "stubbed"
