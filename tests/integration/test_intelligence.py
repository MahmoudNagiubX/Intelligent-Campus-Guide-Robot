"""Integration checks for final intelligence features."""

from __future__ import annotations

import json

from app.pipeline.controller import ConversationController
from app.utils.contracts import IntentClass, TranscriptEvent


class IntelligenceGroq:
    def complete_json(self, *args, **kwargs) -> str:
        text = kwargs.get("user_message", "").lower()
        if "weather" in text:
            intent = IntentClass.OFF_TOPIC.value
            target = ""
        elif "fees" in text:
            intent = IntentClass.ACADEMIC_QUERY.value
            target = "engineering fees"
        else:
            intent = IntentClass.CAMPUS_QUERY.value
            target = "Robotics Lab"
        return json.dumps(
            {
                "intent": intent,
                "confidence": 0.9,
                "language": "en",
                "target_entity": target,
                "clarification_needed": False,
                "clarification_question": "",
            }
        )

    def complete_text(self, *args, **kwargs) -> str:
        message = kwargs.get("user_message", "").lower()
        if "fees" in message:
            return "Engineering and Technology is 56,000 EGP per year."
        return "The Robotics Lab is in Building C, first floor, room C105."


def test_off_topic_question_redirected(monkeypatch):
    groq = IntelligenceGroq()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: groq)
    controller = ConversationController(groq=groq)
    packet = controller.handle_transcript(
        TranscriptEvent(text="what is the weather today", is_final=True, language="en", confidence=0.98)
    )
    assert "outside my expertise" in packet.text
    assert "ECU" in packet.text


def test_academic_question_uses_institutional_path(monkeypatch):
    groq = IntelligenceGroq()
    monkeypatch.setattr("app.routing.router._get_groq", lambda: groq)
    controller = ConversationController(groq=groq)
    packet = controller.handle_transcript(
        TranscriptEvent(text="how much are engineering fees", is_final=True, language="en", confidence=0.98)
    )
    assert "56,000" in packet.text or "EGP" in packet.text
