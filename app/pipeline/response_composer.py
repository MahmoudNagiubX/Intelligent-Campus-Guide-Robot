"""
Navigator - Response Composer

Generates final spoken responses for campus and social paths. Campus answers
are grounded strictly in verified RetrievalResult data and never use the LLM.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.llm.groq_client import GroqClient
from app.utils.contracts import NavigationCommand, ResponsePacket, RetrievalResult, RetrievalStatus, SpokenFacts
from app.utils.logging import get_logger

logger = get_logger(__name__)

_FALLBACK_UNKNOWN_EN = "I am not fully sure about that. I can help with campus locations, staff information, office hours, and navigation."
_FALLBACK_UNKNOWN_AR = "مش متأكد من ده. أقدر أساعدك في أماكن الجامعة، الدكاترة، مواعيدهم، والتنقل."
_FALLBACK_NOT_FOUND_EN = "I could not find that in the campus data. Try asking about a specific room, lab, department, landmark, or staff member."
_FALLBACK_NOT_FOUND_AR = "ملقتش ده في بيانات الجامعة عندي. جرب تسأل عن غرفة أو معمل أو قسم أو معلم أو دكتور معيّن."
_FALLBACK_ERROR_EN = "Something went wrong on my side. Please ask me again."
_FALLBACK_ERROR_AR = "حصلت مشكلة عندي. اسألني تاني."
_FALLBACK_NAV_AMBIGUOUS_EN = "I found a few possible matches. Can you be more specific about where you want to go?"
_FALLBACK_NAV_AMBIGUOUS_AR = "لقيت أكتر من احتمال. ممكن تحدد المكان أكتر؟"
_FALLBACK_SOCIAL_EN = "Hey! How can I help you today?"
_FALLBACK_SOCIAL_AR = "أهلاً! أقدر أساعدك بإيه؟"
_NAV_OFFER_EN = "If you want, I can guide you there."
_NAV_OFFER_AR = "ولو تحب أقدر أوصلك لهناك."


def _is_arabic(language: str) -> bool:
    return language.startswith("ar")


@lru_cache(maxsize=1)
def _load_campus_prompt_en() -> str:
    path = Path("prompts/campus_answer_prompt_en.txt")
    if not path.exists():
        raise FileNotFoundError(f"Campus English prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_campus_prompt_ar() -> str:
    path = Path("prompts/campus_answer_prompt_ar.txt")
    if not path.exists():
        raise FileNotFoundError(f"Campus Arabic prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_social_prompt() -> str:
    path = Path("prompts/social_prompt.txt")
    if not path.exists():
        raise FileNotFoundError(f"Social prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


def _campus_prompt_for(language: str) -> str:
    return _load_campus_prompt_ar() if _is_arabic(language) else _load_campus_prompt_en()


class ResponseComposer:
    """Generate Navigator spoken responses for all intent paths."""

    def __init__(self, groq: GroqClient) -> None:
        self._groq = groq

    def compose_campus_answer(
        self,
        retrieval: RetrievalResult,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        if retrieval.status == RetrievalStatus.NOT_FOUND:
            text = _FALLBACK_NOT_FOUND_AR if _is_arabic(language) else _FALLBACK_NOT_FOUND_EN
            return ResponsePacket(text=text, language=language, session_id=session_id)
        if retrieval.status == RetrievalStatus.AMBIGUOUS:
            return self._compose_clarification(retrieval, language, session_id)
        if retrieval.status == RetrievalStatus.ERROR:
            text = _FALLBACK_ERROR_AR if _is_arabic(language) else _FALLBACK_ERROR_EN
            return ResponsePacket(text=text, language=language, session_id=session_id)

        self._touch_campus_prompt(language)
        spoken = self._render_campus_answer(retrieval, language)
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_navigation_answer(
        self,
        retrieval: RetrievalResult,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        if retrieval.status == RetrievalStatus.AMBIGUOUS:
            text = _FALLBACK_NAV_AMBIGUOUS_AR if _is_arabic(language) else _FALLBACK_NAV_AMBIGUOUS_EN
            return ResponsePacket(text=text, language=language, session_id=session_id)

        if retrieval.status != RetrievalStatus.OK or not retrieval.nav_code:
            packet = self.compose_campus_answer(retrieval, original_query, language, session_id)
            if retrieval.status == RetrievalStatus.OK and retrieval.nav_code is None:
                suffix = (
                    " لسه معنديش مسار موثوق للمكان ده."
                    if _is_arabic(language)
                    else " I do not have a trusted navigation route for that location yet."
                )
                return ResponsePacket(text=packet.text + suffix, language=language, session_id=session_id)
            return packet

        name = retrieval.canonical_name or ("وجهتك" if _is_arabic(language) else "your destination")
        confirmation = (
            f"تمام، هوجّهك لـ {name} دلوقتي."
            if _is_arabic(language)
            else f"Sure! Guiding you to {name} now."
        )
        nav_cmd = NavigationCommand(
            action="navigate",
            target_code=retrieval.nav_code,
            target_label=name,
            spoken_confirmation=confirmation,
            session_id=session_id,
            safety_mode="standard",
        )
        return ResponsePacket(
            text=confirmation,
            language=language,
            should_navigate=True,
            navigation_command=nav_cmd,
            session_id=session_id,
        )

    def compose_social_answer(
        self,
        transcript: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        try:
            prompt = _load_social_prompt()
            raw = self._groq.complete_text(system_prompt=prompt, user_message=transcript, max_tokens=120)
            spoken = self._clean_spoken(raw or "")
        except Exception as exc:
            logger.error("composer_social_llm_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = _FALLBACK_SOCIAL_AR if _is_arabic(language) else _FALLBACK_SOCIAL_EN
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_unknown_answer(self, language: str = "en", session_id: Optional[str] = None) -> ResponsePacket:
        text = _FALLBACK_UNKNOWN_AR if _is_arabic(language) else _FALLBACK_UNKNOWN_EN
        return ResponsePacket(text=text, language=language, session_id=session_id)

    def compose_quality_clarification(
        self,
        *,
        language: str = "en",
        session_id: Optional[str] = None,
        suggestion: Optional[str] = None,
        alternatives: Optional[list[str]] = None,
        ask_location: bool = False,
    ) -> ResponsePacket:
        alternatives = [candidate for candidate in (alternatives or []) if candidate][:2]
        if _is_arabic(language):
            if len(alternatives) >= 2:
                text = f"مش متأكد إني سمعتك صح. تقصد {alternatives[0]} ولا {alternatives[1]}؟"
            elif suggestion and ask_location:
                text = f"تقصد مكان {suggestion}؟"
            elif suggestion:
                text = f"تقصد {suggestion}؟"
            elif ask_location:
                text = "ممكن تحدد المكان اللي تقصده في الجامعة؟"
            else:
                text = "ممكن تعيد السؤال بشكل أوضح شوية؟"
        else:
            if len(alternatives) >= 2:
                text = f"I am not fully sure I heard that correctly. Do you mean {alternatives[0]} or {alternatives[1]}?"
            elif suggestion and ask_location:
                text = f"Are you asking for {suggestion}?"
            elif suggestion:
                text = f"Did you mean {suggestion}?"
            elif ask_location:
                text = "Which campus location do you mean?"
            else:
                text = "Could you say that again a bit more clearly?"
        return ResponsePacket(text=text, language=language, session_id=session_id)

    def _compose_clarification(
        self,
        retrieval: RetrievalResult,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        candidates = retrieval.candidates[:3]
        if not candidates:
            text = _FALLBACK_NOT_FOUND_AR if _is_arabic(language) else _FALLBACK_NOT_FOUND_EN
            return ResponsePacket(text=text, language=language, session_id=session_id)

        candidate_list = ", ".join(f'"{candidate}"' for candidate in candidates)
        if _is_arabic(language):
            text = f"لقيت أكتر من نتيجة: {candidate_list}. تقصد أنهي واحدة؟"
        else:
            text = f"I found a few matches: {candidate_list}. Which one did you mean?"
        return ResponsePacket(text=text, language=language, session_id=session_id)

    def _touch_campus_prompt(self, language: str) -> None:
        try:
            _campus_prompt_for(language)
        except Exception as exc:
            logger.error("composer_campus_prompt_load_error", error=str(exc), language=language)

    def _render_campus_answer(self, retrieval: RetrievalResult, language: str) -> str:
        name = retrieval.canonical_name or ("المكان ده" if _is_arabic(language) else "that place")
        facts = retrieval.spoken_facts
        location_sentence = self._build_location_sentence(name, facts, language)
        details_sentence = self._build_details_sentence(facts, language)
        navigation_sentence = self._build_navigation_offer(retrieval, language)

        parts = [location_sentence]
        if navigation_sentence:
            parts.append(navigation_sentence)
        elif details_sentence:
            parts.append(details_sentence)
        parts = [part for part in parts if part]
        if parts:
            return " ".join(parts[:2]) if len(parts) > 1 else parts[0]
        if _is_arabic(language):
            return f"لقيت {name}."
        return f"I found {name}."

    def _build_location_sentence(self, name: str, facts: Optional[SpokenFacts], language: str) -> str:
        location_bits = self._location_bits(facts, language)
        if not location_bits:
            return f"{name}." if _is_arabic(language) else f"{name}."
        joined = self._join_bits(location_bits, language)
        if _is_arabic(language):
            return f"{name} في {joined}."
        return f"{name} is in {joined}."

    def _build_details_sentence(self, facts: Optional[SpokenFacts], language: str) -> str:
        if not facts:
            return ""

        detail_bits: list[str] = []
        if facts.description:
            detail_bits.append(
                f"الوصف: {facts.description}" if _is_arabic(language) else f"Description: {facts.description}"
            )
        if facts.office_hours:
            detail_bits.append(
                f"مواعيده: {facts.office_hours}" if _is_arabic(language) else f"Office hours are {facts.office_hours}"
            )
        if facts.contact_notes:
            detail_bits.append(
                f"ملاحظات التواصل: {facts.contact_notes}"
                if _is_arabic(language)
                else f"Contact notes: {facts.contact_notes}"
            )

        if not detail_bits:
            return ""
        return f"{self._join_bits(detail_bits, language)}."

    def _build_navigation_offer(self, retrieval: RetrievalResult, language: str) -> str:
        if not retrieval.nav_code:
            return ""
        return _NAV_OFFER_AR if _is_arabic(language) else _NAV_OFFER_EN

    def _location_bits(self, facts: Optional[SpokenFacts], language: str) -> list[str]:
        if not facts:
            return []

        bits: list[str] = []
        if facts.building:
            bits.append(f"المبنى {facts.building}" if _is_arabic(language) else f"building {facts.building}")
        if facts.floor:
            bits.append(f"الدور {facts.floor}" if _is_arabic(language) else f"floor {facts.floor}")
        if facts.room:
            bits.append(f"الغرفة {facts.room}" if _is_arabic(language) else f"room {facts.room}")
        return bits

    @staticmethod
    def _join_bits(bits: list[str], language: str) -> str:
        separator = "، " if _is_arabic(language) else ", "
        return separator.join(bit for bit in bits if bit)

    @staticmethod
    def _clean_spoken(raw: str) -> str:
        text = raw.strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
                for key in ("text", "response", "answer", "reply", "message"):
                    if key in data:
                        return str(data[key]).strip()
            except json.JSONDecodeError:
                pass
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        return text
