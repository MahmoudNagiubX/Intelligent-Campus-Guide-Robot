"""
Arabic transcript normalization for retrieval matching.

Applied before routing and retrieval. Do not apply this to TTS output.
Latin substrings, room numbers, and building codes are preserved.
"""

from __future__ import annotations

import re

_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF]")
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_TATWEEL = "\u0640"
_ALEF_VARIANTS = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0671": "\u0627",
        "\u0649": "\u064A",
    }
)
_SPOKEN_VARIANTS: dict[str, str] = {
    "فين": "اين",
    "وين": "اين",
    "امتى": "متى",
    "ازاي": "كيف",
    "ازيك": "كيف حالك",
    "ايه": "ما",
    "اايه": "ما",
    "عايز": "اريد",
    "عاوز": "اريد",
    "عوزه": "اريده",
    "عوزها": "اريدها",
    "حابب": "اريد",
    "مش": "لا",
    "بتاع": "خاص بـ",
    "دلوقتي": "الان",
    "دلوقت": "الان",
    "اللي": "الذي",
    "اللى": "الذي",
    "اوضة": "غرفة",
    "اوضه": "غرفة",
    "اوده": "غرفة",
    "رووم": "غرفة",
    "مختبر": "معمل",
    "لاب": "معمل",
}
_SPOKEN_PHRASES: tuple[tuple[str, str], ...] = (
    ("مش عارف", "لا اعرف"),
)
_ROOM_ARABIC_PREFIXES = re.compile(
    r"(?:اوضة|اوضه|غرفة|غرفه|اوده|رووم|رقم\s*الغرفة)\s*",
    re.IGNORECASE,
)
_ROOM_ENGLISH_PREFIXES = re.compile(
    r"\b(?:room\s*no\.?\s*|room(?=\d)|r(?=\d)|lab\s*(?=\d))",
    re.IGNORECASE,
)


def normalize_arabic_transcript(text: str) -> str:
    """
    Normalize Arabic STT output for retrieval matching while preserving Latin segments.
    """
    if not text:
        return text

    result_tokens: list[str] = []
    for token, is_arabic in _split_preserve_latin(text):
        if not is_arabic:
            result_tokens.append(token)
            continue
        value = token.replace(_TATWEEL, "")
        value = _DIACRITICS.sub("", value)
        value = value.translate(_ALEF_VARIANTS)
        value = _apply_spoken_variants(value)
        value = value.translate(_ALEF_VARIANTS)
        result_tokens.append(value)

    return " ".join("".join(result_tokens).split())


def normalize_arabic_for_storage(text: str) -> str:
    """Normalize Arabic for FTS storage, including teh marbuta broad matching."""
    value = normalize_arabic_transcript(text)
    return value.replace("\u0629", "\u0647")


def normalize_room_reference(text: str) -> str:
    """Normalize common English and Arabic room references to ``room NNN``."""
    value = (text or "").strip()
    value = _ROOM_ARABIC_PREFIXES.sub("room ", value)
    value = _ROOM_ENGLISH_PREFIXES.sub("room ", value)
    value = re.sub(r"\broom\s+(\d+)\b", r"room \1", value, flags=re.IGNORECASE)
    return " ".join(value.split())


def _split_preserve_latin(text: str) -> list[tuple[str, bool]]:
    """Split text into normalizable Arabic segments and preserved non-Arabic segments."""
    tokens: list[tuple[str, bool]] = []
    buf = ""
    in_arabic = False

    for ch in text:
        ch_is_arabic = bool(_ARABIC_CHARS.match(ch))
        if ch == " ":
            if buf:
                tokens.append((buf, in_arabic))
                buf = ""
            tokens.append((" ", False))
            in_arabic = False
        elif ch_is_arabic == in_arabic:
            buf += ch
        else:
            if buf:
                tokens.append((buf, in_arabic))
            buf = ch
            in_arabic = ch_is_arabic

    if buf:
        tokens.append((buf, in_arabic))
    return tokens


def _apply_spoken_variants(text: str) -> str:
    for source, target in _SPOKEN_PHRASES:
        text = text.replace(source, target)
    words = text.split()
    return " ".join(_SPOKEN_VARIANTS.get(word, word) for word in words)
