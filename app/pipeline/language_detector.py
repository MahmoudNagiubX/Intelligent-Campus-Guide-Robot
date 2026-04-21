"""
Navigator - Language Detector

Two-layer detection:
  Layer 1: trust Deepgram when it reports language with enough confidence.
  Layer 2: use a Unicode Arabic-script heuristic as a fast fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


@dataclass(frozen=True)
class LangResult:
    code: str
    source: str
    confidence: float


def detect_language(
    text: str,
    deepgram_lang: str | None = None,
    deepgram_confidence: float | None = None,
    confidence_threshold: float = 0.80,
    default_lang: str = "en",
) -> LangResult:
    """Determine the language of one transcript."""
    if deepgram_lang:
        provider_conf = deepgram_confidence if deepgram_confidence is not None else 1.0
        if provider_conf >= confidence_threshold:
            return LangResult(
                code=_normalise_lang_code(deepgram_lang),
                source="stt_provider",
                confidence=provider_conf,
            )

    if text:
        compact = "".join(ch for ch in text if not ch.isspace())
        letter_like = "".join(ch for ch in compact if ch.isalpha() or _ARABIC_PATTERN.match(ch))
        if letter_like:
            arabic_chars = len(_ARABIC_PATTERN.findall(letter_like))
            ratio = arabic_chars / len(letter_like)
            if ratio >= 0.15:
                return LangResult(code="ar", source="unicode_heuristic", confidence=min(0.95, 0.5 + ratio))
            if ratio < 0.05:
                return LangResult(code="en", source="unicode_heuristic", confidence=0.90)

    return LangResult(code=default_lang, source="default", confidence=0.50)


def _normalise_lang_code(code: str) -> str:
    normalized = code.strip().lower()
    if normalized.startswith("ar"):
        return "ar-EG" if "eg" in normalized else "ar"
    if normalized.startswith("en"):
        return "en"
    return normalized


def lang_is_arabic(lang: LangResult) -> bool:
    return lang.code.startswith("ar")


def lang_is_english(lang: LangResult) -> bool:
    return lang.code.startswith("en")
