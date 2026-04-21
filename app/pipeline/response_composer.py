"""
Navigator - Response Composer

Generates final spoken responses for campus and social paths. Campus answers
use Groq to rewrite verified retrieval facts into short natural speech.
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
_NAV_OFFER_EN = "I can guide you there if you'd like."
_NAV_OFFER_AR = "أقدر آخدك هناك لو حابب."

_ENTITY_TYPE_LABEL_EN = {
    "room": "Room",
    "lab": "Lab",
    "department": "Department",
    "landmark": "Landmark",
    "staff": "Staff",
}
_ENTITY_TYPE_LABEL_AR = {
    "room": "غرفة",
    "lab": "معمل",
    "department": "قسم",
    "landmark": "معلم",
    "staff": "دكتور",
}
_DESCRIPTION_TRANSLATIONS_AR = {
    "lab": "معمل",
    "office": "مكتب",
    "administration": "إدارة",
    "department": "قسم",
    "landmark": "معلم",
    "room": "غرفة",
    "open": "مفتوح",
    "closed": "مغلق",
}


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

        facts_block = self._format_facts_block(retrieval, language)
        prompt = self._render_prompt_template(_campus_prompt_for(language), facts_block)
        try:
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=facts_block,
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "")
        except Exception as exc:
            logger.error("composer_campus_llm_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = self._fallback_from_facts(retrieval, language)

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

    def _format_facts_block(self, retrieval: RetrievalResult, language: str) -> str:
        facts = []
        facts_map = self._facts_map(retrieval, language)
        for label, value in facts_map:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            facts.append(f"{label}: {text}")
        return "\n".join(facts)

    @staticmethod
    def _render_prompt_template(prompt: str, facts_block: str) -> str:
        rendered = prompt.replace("{facts}", facts_block)
        rendered = rendered.replace("{retrieval_facts}", facts_block)
        return rendered

    def _facts_map(self, retrieval: RetrievalResult, language: str) -> list[tuple[str, Optional[str]]]:
        facts = retrieval.spoken_facts
        if _is_arabic(language):
            return [
                ("الاسم", retrieval.canonical_name),
                ("المبنى", self._fact_value(facts, "building")),
                ("الدور", self._fact_value(facts, "floor")),
                ("الغرفة", self._fact_value(facts, "room")),
                ("النوع", self._entity_type_label(retrieval.entity_type, language)),
                ("الوصف", self._description_value(facts, language)),
                ("الساعات", self._fact_value(facts, "office_hours")),
                ("ملاحظات التواصل", self._fact_value(facts, "contact_notes")),
                ("رمز التوجيه", retrieval.nav_code),
            ]
        return [
            ("Name", retrieval.canonical_name),
            ("Building", self._fact_value(facts, "building")),
            ("Floor", self._fact_value(facts, "floor")),
            ("Room", self._fact_value(facts, "room")),
            ("Type", self._entity_type_label(retrieval.entity_type, language)),
            ("Description", self._description_value(facts, language)),
            ("Office hours", self._fact_value(facts, "office_hours")),
            ("Contact notes", self._fact_value(facts, "contact_notes")),
            ("Nav code", retrieval.nav_code),
        ]

    @staticmethod
    def _fact_value(facts: Optional[SpokenFacts], field_name: str) -> Optional[str]:
        if facts is None:
            return None
        value = getattr(facts, field_name, None)
        return str(value) if value is not None else None

    def _entity_type_label(self, entity_type: Optional[str], language: str) -> Optional[str]:
        if not entity_type:
            return None
        if _is_arabic(language):
            return _ENTITY_TYPE_LABEL_AR.get(entity_type, entity_type)
        return _ENTITY_TYPE_LABEL_EN.get(entity_type, entity_type.replace("_", " ").title())

    def _description_value(self, facts: Optional[SpokenFacts], language: str) -> Optional[str]:
        raw = self._fact_value(facts, "description")
        if raw is None:
            return None
        if _is_arabic(language):
            return self._translate_term_ar(raw)
        return raw

    def _translate_term_ar(self, value: str) -> str:
        normalized = value.strip()
        translated = _DESCRIPTION_TRANSLATIONS_AR.get(normalized.lower())
        return translated or normalized

    def _fallback_from_facts(self, retrieval: RetrievalResult, language: str) -> str:
        name = retrieval.canonical_name or ("المكان ده" if _is_arabic(language) else "that place")
        facts = retrieval.spoken_facts
        if _is_arabic(language):
            parts = []
            if facts and facts.building:
                parts.append(f"في المبنى {facts.building}")
            if facts and facts.floor:
                parts.append(f"في الدور {facts.floor}")
            if facts and facts.room:
                parts.append(f"غرفة {facts.room}")
            location = " ".join(parts).strip()
            sentence = f"{name} {location}.".strip() if location else f"لقيت {name}."
            if retrieval.nav_code:
                return f"{sentence} {_NAV_OFFER_AR}"
            return sentence

        parts = []
        if facts and facts.building:
            parts.append(f"in building {facts.building}")
        if facts and facts.floor:
            parts.append(f"on floor {facts.floor}")
        if facts and facts.room:
            parts.append(f"room {facts.room}")
        location = ", ".join(parts)
        sentence = f"{name} is {location}." if location else f"I found {name}."
        if retrieval.nav_code:
            return f"{sentence} {_NAV_OFFER_EN}"
        return sentence

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
