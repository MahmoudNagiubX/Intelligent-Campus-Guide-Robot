from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.routing.router import _apply_pre_rules, _parse_router_response, route
from app.utils.contracts import IntentClass


def _router_payload(**overrides) -> str:
    payload = {
        "intent": "Campus_Query",
        "confidence": 0.94,
        "language": "ar",
        "needs_retrieval": True,
        "needs_navigation": False,
        "target_entity": "معمل الروبوتات",
        "target_type": "room",
        "normalized_query": "فين معمل الروبوتات",
        "clarification_needed": False,
        "clarification_question": "",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_pre_rules_cover_english_and_arabic_paths() -> None:
    assert _apply_pre_rules("take me to C105") == IntentClass.NAVIGATION_REQUEST
    assert _apply_pre_rules("where is the robotics lab") == IntentClass.CAMPUS_QUERY
    assert _apply_pre_rules("ازيك") == IntentClass.SOCIAL_CHAT


def test_parse_router_response_maps_bilingual_prompt_contract() -> None:
    result = _parse_router_response(_router_payload(), "فين معمل الروبوتات", fallback_language="en")

    assert result.intent == IntentClass.CAMPUS_QUERY
    assert result.language == "ar"
    assert result.target_text == "معمل الروبوتات"
    assert result.needs_clarification is False


def test_parse_router_response_accepts_clarification_needed_field() -> None:
    result = _parse_router_response(
        _router_payload(
            intent="Navigation_Request",
            clarification_needed=True,
            clarification_question="تقصد أي معمل؟",
            target_entity="",
        ),
        "وديني المعمل",
        fallback_language="ar",
    )

    assert result.intent == IntentClass.NAVIGATION_REQUEST
    assert result.needs_clarification is True
    assert result.clarification_question == "تقصد أي معمل؟"
    assert result.target_text is None


def test_route_uses_llm_json_and_lang_hint() -> None:
    groq = MagicMock()
    groq.complete_json.return_value = _router_payload(
        intent="Social_Chat",
        language="en",
        needs_retrieval=False,
        target_entity="",
    )

    with patch("app.routing.router._get_groq", return_value=groq):
        result = route("hello navigator", lang_hint="en")

    assert result.intent == IntentClass.SOCIAL_CHAT
    assert result.language == "en"


def test_route_falls_back_to_pre_rule_when_llm_fails() -> None:
    groq = MagicMock()
    groq.complete_json.return_value = None

    with patch("app.routing.router._get_groq", return_value=groq):
        result = route("take me to the robotics lab", lang_hint="en")

    assert result.intent == IntentClass.NAVIGATION_REQUEST
    assert result.confidence >= 0.7


def test_route_returns_unknown_for_empty_transcript() -> None:
    result = route("   ", lang_hint="ar")

    assert result.intent == IntentClass.UNKNOWN
    assert result.language == "ar"
