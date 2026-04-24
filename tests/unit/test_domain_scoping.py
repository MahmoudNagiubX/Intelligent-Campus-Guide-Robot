"""Test that Off_Topic and Academic_Query intents are available and parsed."""


def test_off_topic_intent_exists():
    from app.utils.contracts import IntentClass

    assert IntentClass.OFF_TOPIC == "off_topic"


def test_academic_query_intent_exists():
    from app.utils.contracts import IntentClass

    assert IntentClass.ACADEMIC_QUERY == "academic_query"


def test_off_topic_intent_parsing():
    from app.routing.router import _parse_intent
    from app.utils.contracts import IntentClass

    assert _parse_intent("off_topic") == IntentClass.OFF_TOPIC
    assert _parse_intent("Off_Topic") == IntentClass.OFF_TOPIC
    assert _parse_intent("offtopic") == IntentClass.OFF_TOPIC


def test_academic_query_intent_parsing():
    from app.routing.router import _parse_intent
    from app.utils.contracts import IntentClass

    assert _parse_intent("academic_query") == IntentClass.ACADEMIC_QUERY
    assert _parse_intent("Academic_Query") == IntentClass.ACADEMIC_QUERY

