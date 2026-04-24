"""
Navigator - Response Composer

Generates final spoken responses for campus and social paths. Campus answers
use Groq to rewrite verified retrieval facts into short natural speech.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.llm.groq_client import GroqClient
from app.retrieval.ecu_knowledge import ECUKnowledgeResult
from app.retrieval.ecu_knowledge_ar import ECUKnowledgeArResult
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
_ARABIC_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")
_FALLBACK_SOCIAL_AR = "أهلاً! أقدر أساعدك بإيه؟"
_NAV_OFFER_EN = "I can guide you there if you'd like."
_NAV_OFFER_AR = "أقدر آخدك هناك لو حابب."
_ROBOTIC_PREAMBLES_EN = [
    "certainly! ",
    "certainly, ",
    "of course! ",
    "of course, ",
    "sure! ",
    "sure, ",
    "absolutely! ",
    "absolutely, ",
    "great question! ",
    "great question, ",
    "i'd be happy to help! ",
    "i'd be happy to help, ",
    "based on the data",
    "according to my records",
    "according to the information",
    "based on the information",
    "i've found that ",
    "i can see that ",
    "it appears that ",
]
_ROBOTIC_PREAMBLES_AR = [
    "بالتأكيد،",
    "بالتأكيد",
    "أكيد،",
    "أكيد",
    "يسعدني",
    "بناءً على البيانات",
    "وفقاً للسجلات",
    "وفقًا للسجلات",
    "كمساعد ذكاء اصطناعي",
]

_ENTITY_TYPE_LABEL_EN = {
    "room": "Room",
    "lab": "Lab",
    "department": "Department",
    "landmark": "Landmark",
    "staff": "Staff",
    "building": "Building",
    "member": "Member",
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


def _contains_arabic_script(text: str) -> bool:
    """Return True when text contains Arabic Unicode characters."""
    return bool(_ARABIC_SCRIPT_PATTERN.search(text or ""))


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


@lru_cache(maxsize=1)
def _load_ecu_prompt_en() -> str:
    path = Path("prompts/ecu_answer_prompt_en.txt")
    if not path.exists():
        raise FileNotFoundError(f"ECU English prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_general_campus_prompt_en() -> str:
    path = Path("prompts/general_campus_prompt_en.txt")
    if not path.exists():
        raise FileNotFoundError(f"General campus English prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_academic_prompt_en() -> str:
    path = Path("prompts/academic_answer_prompt_en.txt")
    if not path.exists():
        raise FileNotFoundError(f"Academic prompt missing: {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_ecu_prompt_ar() -> str:
    path = Path("prompts/ecu_answer_prompt_ar.txt")
    if not path.exists():
        raise FileNotFoundError(f"ECU Arabic prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_general_campus_prompt_ar() -> str:
    path = Path("prompts/general_campus_prompt_ar.txt")
    if not path.exists():
        raise FileNotFoundError(f"General campus Arabic prompt not found at {path}")
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

        if _is_arabic(language):
            facts_block = self._format_facts_block(retrieval, language)
        else:
            from app.retrieval.context_builder import build_rich_context

            facts_block = build_rich_context(retrieval) or self._format_facts_block(retrieval, language)
        prompt = self._render_prompt_template(_campus_prompt_for(language), facts_block, question=original_query)
        try:
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=f"Question: {original_query}\n\nFacts:\n{facts_block}",
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "", language)
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
        base_prompt = _load_social_prompt()
        if _is_arabic(language):
            lang_instruction = (
                "ABSOLUTE RULE: The user spoke ARABIC. "
                "You MUST reply in Egyptian Arabic ONLY. "
                "Never reply in English when this instruction is present.\n\n"
            )
        else:
            lang_instruction = (
                "ABSOLUTE RULE: The user spoke ENGLISH. "
                "You MUST reply in ENGLISH ONLY. "
                "Never reply in Arabic or any other language when this instruction is present.\n\n"
            )
        prompt = lang_instruction + base_prompt

        try:
            raw = self._groq.complete_text(system_prompt=prompt, user_message=transcript, max_tokens=120)
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer_social_llm_error", error=str(exc))
            spoken = ""

        if spoken and _is_arabic(language) and not _contains_arabic_script(spoken):
            logger.warning("composer_social_wrong_language", expected="ar", got="en", text=spoken[:40])
            spoken = _FALLBACK_SOCIAL_AR
        if spoken and not _is_arabic(language) and _contains_arabic_script(spoken):
            logger.warning("composer_social_wrong_language", expected="en", got="ar", text=spoken[:40])
            spoken = _FALLBACK_SOCIAL_EN

        if not spoken:
            spoken = _FALLBACK_SOCIAL_AR if _is_arabic(language) else _FALLBACK_SOCIAL_EN
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_unknown_answer(self, language: str = "en", session_id: Optional[str] = None) -> ResponsePacket:
        text = _FALLBACK_UNKNOWN_AR if _is_arabic(language) else _FALLBACK_UNKNOWN_EN
        return ResponsePacket(text=text, language=language, session_id=session_id)

    def compose_ecu_answer(
        self,
        ecu_result: ECUKnowledgeResult,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Compose an English response from the local ECU website cache."""
        if _is_arabic(language):
            return self.compose_unknown_answer(language=language, session_id=session_id)

        context = self._format_ecu_context(ecu_result)
        prompt = _load_ecu_prompt_en().replace("{context}", context)
        try:
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=f"Question: {original_query}\n\nContext:\n{context}",
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer_ecu_llm_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = ecu_result.content or "I found relevant ECU information, but I do not have a concise detail to say yet."
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_academic_answer(
        self,
        routing_text: str,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Answer academic/institutional questions using ECU institutional knowledge."""
        if _is_arabic(language):
            return self.compose_unknown_answer(language=language, session_id=session_id)

        from app.retrieval.ecu_institutional import build_institutional_context

        context = build_institutional_context(routing_text or original_query)
        prompt = _load_academic_prompt_en().replace("{context}", context)
        try:
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=original_query,
                max_tokens=250,
            )
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer.academic_answer_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = "I don't have that exact detail. Check ecu.edu.eg or ask the registrar."
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_arabic_hybrid_answer(
        self,
        hybrid: "ArabicHybridResult",
        original_query: str,
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Compose an Arabic response from DB, Arabic ECU cache, or general fallback."""
        language = "ar-EG"
        if hybrid.answered_by in ("db", "clarification") and hybrid.db_result is not None:
            return self.compose_campus_answer(hybrid.db_result, original_query, language, session_id)
        if hybrid.answered_by == "ecu_web" and hybrid.ecu_result is not None:
            return self._compose_arabic_ecu_answer(hybrid.ecu_result, original_query, session_id)
        return self._compose_arabic_general_answer(original_query, session_id)

    def _compose_arabic_ecu_answer(
        self,
        ecu_result: ECUKnowledgeArResult,
        original_query: str,
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Compose an Egyptian-Arabic response from the local Arabic ECU cache."""
        language = "ar-EG"
        context = self._format_ecu_ar_context(ecu_result)
        prompt = _load_ecu_prompt_ar().replace("{context}", context).replace("{question}", original_query)
        try:
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=f"السؤال: {original_query}\n\nالمعلومات:\n{context}",
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer_arabic_ecu_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = ecu_result.content or "معنديش تفاصيل كفاية عن ده دلوقتي. جرب ecu.edu.eg أو شؤون الطلاب."
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def _compose_arabic_general_answer(
        self,
        original_query: str,
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Compose an honest Egyptian-Arabic general campus answer without retrieval facts."""
        language = "ar-EG"
        try:
            raw = self._groq.complete_text(
                system_prompt=_load_general_campus_prompt_ar(),
                user_message=original_query,
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer_arabic_general_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = "مش عندي المعلومة دي دلوقتي. جرب تسأل شؤون الطلاب أو راجع ecu.edu.eg."
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    def compose_general_campus_answer(
        self,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Compose an honest English campus response without DB facts."""
        if _is_arabic(language):
            return self.compose_unknown_answer(language=language, session_id=session_id)

        try:
            raw = self._groq.complete_text(
                system_prompt=_load_general_campus_prompt_en(),
                user_message=original_query,
                max_tokens=180,
            )
            spoken = self._clean_spoken(raw or "", language)
        except Exception as exc:
            logger.error("composer_general_campus_llm_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = "I don't have that specific detail. You can check ecu.edu.eg or ask at the student affairs office."
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

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
    def _render_prompt_template(prompt: str, facts_block: str, question: str = "") -> str:
        rendered = prompt.replace("{facts}", facts_block)
        rendered = rendered.replace("{retrieval_facts}", facts_block)
        rendered = rendered.replace("{question}", question)
        rendered = rendered.replace("{user_question}", question)
        return rendered

    @staticmethod
    def _format_ecu_context(ecu_result: ECUKnowledgeResult) -> str:
        parts = [
            ("Title", ecu_result.title),
            ("Content", ecu_result.content),
            ("Source", ecu_result.source_url),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)

    @staticmethod
    def _format_ecu_ar_context(ecu_result: ECUKnowledgeArResult) -> str:
        parts = [
            ("العنوان", ecu_result.title),
            ("المحتوى", ecu_result.content),
            ("المصدر", ecu_result.source_url),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)

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
            ("Title", self._fact_value(facts, "title")),
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
    def _clean_spoken(raw: str, language: str = "en") -> str:
        text = raw.strip()
        if text.startswith("{"):
            try:
                data = json.loads(text)
                for key in ("text", "response", "answer", "reply", "message"):
                    if key in data:
                        text = str(data[key]).strip()
                        break
            except json.JSONDecodeError:
                pass
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()
        if _is_arabic(language):
            text = _clean_spoken_ar(text)
        else:
            text = _clean_spoken_en(text)
        return text


def _clean_spoken_en(text: str) -> str:
    """Remove robotic preambles and cleanup English TTS text."""
    cleaned = text.strip()
    lowered = cleaned.lower()
    for preamble in _ROBOTIC_PREAMBLES_EN:
        if lowered.startswith(preamble):
            cleaned = cleaned[len(preamble) :].lstrip(", ")
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:]
            break
    cleaned = re.sub(
        r"\s+(?:is there anything else|can i help you with anything else|let me know if you need|feel free to ask)[^.]*\.?$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _clean_spoken_ar(text: str) -> str:
    """Remove robotic Arabic preambles while preserving Egyptian wording."""
    cleaned = text.strip()
    for preamble in _ROBOTIC_PREAMBLES_AR:
        if cleaned.startswith(preamble):
            cleaned = cleaned[len(preamble) :].lstrip("،, ")
            break
    cleaned = re.sub(r"\s+(?:هل تحتاج أي مساعدة أخرى|هل أقدر أساعدك في حاجة تانية)[؟?]?$", "", cleaned).strip()
    return cleaned
