from __future__ import annotations

from app.pipeline.language_detector import detect_language, lang_is_arabic, lang_is_english


def test_detect_language_trusts_deepgram_when_confident() -> None:
    result = detect_language("where is the robotics lab", deepgram_lang="en", deepgram_confidence=0.97)

    assert result.code == "en"
    assert result.source == "stt_provider"
    assert lang_is_english(result) is True


def test_detect_language_falls_back_to_arabic_unicode_heuristic() -> None:
    result = detect_language("فين معمل الروبوتات", deepgram_lang=None)

    assert result.code == "ar"
    assert result.source == "unicode_heuristic"
    assert lang_is_arabic(result) is True


def test_detect_language_falls_back_to_english_unicode_heuristic() -> None:
    result = detect_language("where is the robotics lab", deepgram_lang=None)

    assert result.code == "en"
    assert result.source == "unicode_heuristic"


def test_detect_language_uses_default_when_all_signals_are_weak() -> None:
    result = detect_language("12345", deepgram_lang=None, default_lang="ar")

    assert result.code == "ar"
    assert result.source == "default"
