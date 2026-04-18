"""
Navigator - Response Composer
Phase 5 + Phase 6

Generates final spoken responses for the campus and social paths.
All campus answers are grounded strictly in verified RetrievalResult data.
Social answers are kept short and persona-consistent.

Rules:
- Campus path: ONLY use facts from the RetrievalResult — never invent.
- Social path: Use the social prompt, redirect campus facts back to truth layer.
- Unknown path: Return a fixed bounded fallback (no LLM call needed).
- All responses must be short and TTS-friendly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.llm.groq_client import GroqClient
from app.utils.contracts import (
    IntentClass,
    RetrievalResult,
    RetrievalStatus,
    ResponsePacket,
    NavigationCommand,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

# ── Bounded fallbacks (no LLM) ────────────────────────────────────────────────

_FALLBACK_UNKNOWN_EN = (
    "I'm not fully sure about that. I can help with campus locations, "
    "staff information, office hours, and navigation."
)
_FALLBACK_UNKNOWN_AR = (
    "مش متأكد من ده. أنا بساعد في مواقع الجامعة، معلومات أعضاء هيئة التدريس، "
    "المواعيد، والتوجيه."
)

_FALLBACK_NOT_FOUND_EN = (
    "I couldn't find that in my campus database. "
    "Try rephrasing or ask me about a specific location, staff member, or department."
)
_FALLBACK_NOT_FOUND_AR = (
    "ماعنديش معلومات عن ده في قاعدة بياناتي. "
    "جرب تسأل عن مكان، دكتور، أو قسم معين."
)

_FALLBACK_ERROR_EN = (
    "Something went wrong on my end. Let me reset — please ask your question again."
)

_FALLBACK_NAV_AMBIGUOUS_EN = (
    "I found a few possible matches. Can you be more specific about where you want to go?"
)
_FALLBACK_NAV_AMBIGUOUS_AR = (
    "لقيت أكتر من نتيجة. ممكن توضح أكتر؟"
)

_CLARIFICATION_PREFIX_EN = "I found a few matches: "
_CLARIFICATION_PREFIX_AR = "لقيت أكتر من واحد: "


# ── Prompt loaders ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_campus_prompt() -> str:
    path = Path("prompts/campus_answer_prompt.txt")
    if not path.exists():
        raise FileNotFoundError(f"Campus answer prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=1)
def _load_social_prompt() -> str:
    path = Path("prompts/social_prompt.txt")
    if not path.exists():
        raise FileNotFoundError(f"Social prompt not found at {path}")
    return path.read_text(encoding="utf-8").strip()


# ── Main composer ─────────────────────────────────────────────────────────────

class ResponseComposer:
    """
    Generates Navigator's spoken responses for all intent paths.

    Args:
        groq: A GroqClient instance (shared with the router).
    """

    def __init__(self, groq: GroqClient) -> None:
        self._groq = groq

    # ── Campus path ───────────────────────────────────────────────────────────

    def compose_campus_answer(
        self,
        retrieval: RetrievalResult,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """
        Build a spoken campus answer from a verified RetrievalResult.
        NEVER invents facts — only uses what retrieval returned.
        """
        if retrieval.status == RetrievalStatus.NOT_FOUND:
            text = _FALLBACK_NOT_FOUND_EN if language == "en" else _FALLBACK_NOT_FOUND_AR
            return ResponsePacket(text=text, language=language, session_id=session_id)

        if retrieval.status == RetrievalStatus.AMBIGUOUS:
            return self._compose_clarification(retrieval, language, session_id)

        if retrieval.status == RetrievalStatus.ERROR:
            return ResponsePacket(text=_FALLBACK_ERROR_EN, language=language, session_id=session_id)

        # Build a facts JSON to pass to the LLM
        facts = self._build_facts_dict(retrieval, original_query)
        facts_json = json.dumps(facts, ensure_ascii=False)

        try:
            prompt = _load_campus_prompt()
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=facts_json,
                max_tokens=200,
            )
            # The campus prompt asks for plain text, not JSON.
            # complete_json returns the raw string — which here is the spoken answer.
            spoken = self._clean_spoken(raw or "")
        except Exception as exc:
            logger.error("composer_campus_llm_error", error=str(exc))
            spoken = self._fallback_from_facts(retrieval, language)

        if not spoken:
            spoken = self._fallback_from_facts(retrieval, language)

        logger.info(
            "composer_campus_answer",
            entity=retrieval.canonical_name,
            language=language,
            spoken_preview=spoken[:80],
        )

        return ResponsePacket(
            text=spoken,
            language=language,
            session_id=session_id,
        )

    # ── Navigation path ───────────────────────────────────────────────────────

    def compose_navigation_answer(
        self,
        retrieval: RetrievalResult,
        original_query: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """
        Build a spoken navigation confirmation and construct the action command.
        If the target cannot be resolved, block movement and explain why.
        """
        if retrieval.status == RetrievalStatus.AMBIGUOUS:
            text = (
                _FALLBACK_NAV_AMBIGUOUS_AR if language == "ar-EG"
                else _FALLBACK_NAV_AMBIGUOUS_EN
            )
            return ResponsePacket(text=text, language=language, session_id=session_id)

        if retrieval.status != RetrievalStatus.OK or not retrieval.nav_code:
            # No valid navigation target — answer factually but don't move
            packet = self.compose_campus_answer(retrieval, original_query, language, session_id)
            if retrieval.nav_code is None and retrieval.status == RetrievalStatus.OK:
                # We know the location but have no nav code
                suffix_en = " I don't have a navigation route for that location yet."
                suffix_ar = " مش عندي مسار توجيه لده لحد دلوقتي."
                suffix = suffix_ar if language == "ar-EG" else suffix_en
                return ResponsePacket(
                    text=packet.text + suffix,
                    language=language,
                    session_id=session_id,
                )
            return packet

        # Build confirmation text
        name = retrieval.canonical_name or "your destination"
        if language == "ar-EG":
            confirmation = f"تمام، هوديك على {name} دلوقتي!"
        else:
            confirmation = f"Sure! Guiding you to the {name} now."

        nav_cmd = NavigationCommand(
            action="navigate",
            target_code=retrieval.nav_code,
            target_label=name,
            spoken_confirmation=confirmation,
            session_id=session_id,
            safety_mode="standard",
        )

        logger.info(
            "composer_navigation_command",
            target_code=retrieval.nav_code,
            target_label=name,
            language=language,
        )

        return ResponsePacket(
            text=confirmation,
            language=language,
            should_navigate=True,
            navigation_command=nav_cmd,
            session_id=session_id,
        )

    # ── Social path ───────────────────────────────────────────────────────────

    def compose_social_answer(
        self,
        transcript: str,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """
        Generate a short, warm persona response for social chat.
        No campus facts are used here.
        """
        try:
            prompt = _load_social_prompt()
            raw = self._groq.complete_text(
                system_prompt=prompt,
                user_message=transcript,
                max_tokens=150,
            )
            spoken = self._clean_spoken(raw or "")
        except Exception as exc:
            logger.error("composer_social_llm_error", error=str(exc))
            spoken = ""

        if not spoken:
            spoken = (
                "مرحباً! كيف أقدر أساعدك؟" if language == "ar-EG"
                else "Hey! How can I help you today?"
            )

        logger.info("composer_social_answer", language=language, preview=spoken[:60])
        return ResponsePacket(text=spoken, language=language, session_id=session_id)

    # ── Unknown path ──────────────────────────────────────────────────────────

    def compose_unknown_answer(
        self,
        language: str = "en",
        session_id: Optional[str] = None,
    ) -> ResponsePacket:
        """Return a fixed, bounded fallback. No LLM call."""
        text = _FALLBACK_UNKNOWN_AR if language == "ar-EG" else _FALLBACK_UNKNOWN_EN
        logger.info("composer_unknown_fallback", language=language)
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
        """Return a short clarification when transcript quality or evidence is weak."""
        alts = [candidate for candidate in (alternatives or []) if candidate][:2]

        if language == "ar-EG":
            if len(alts) >= 2:
                text = f"مش متأكد إني سمعتك صح. تقصد {alts[0]} ولا {alts[1]}؟"
            elif suggestion and ask_location:
                text = f"هل تقصد موقع {suggestion}؟"
            elif suggestion:
                text = f"هل تقصد {suggestion}؟"
            elif ask_location:
                text = "مش متأكد إني سمعت المكان صح. تقصد أي موقع في الجامعة؟"
            else:
                text = "مش متأكد إني سمعتك صح. ممكن تعيد سؤالك باختصار؟"
        else:
            if len(alts) >= 2:
                text = f"I'm not fully sure I heard that correctly. Do you mean the {alts[0]} or {alts[1]}?"
            elif suggestion and ask_location:
                text = f"Are you asking for the {suggestion} location?"
            elif suggestion:
                text = f"Did you mean the {suggestion}?"
            elif ask_location:
                text = "I'm not fully sure I heard the location correctly. Which campus location do you mean?"
            else:
                text = "I'm not fully sure I heard that correctly. Could you say it again a bit more clearly?"

        logger.info(
            "composer_quality_clarification",
            language=language,
            suggestion=suggestion,
            alternatives=alts,
            ask_location=ask_location,
        )
        return ResponsePacket(text=text, language=language, session_id=session_id)

    # ── Clarification ─────────────────────────────────────────────────────────

    def _compose_clarification(
        self,
        retrieval: RetrievalResult,
        language: str,
        session_id: Optional[str],
    ) -> ResponsePacket:
        """Build a clarification question from ambiguous candidates."""
        candidates = retrieval.candidates[:3]
        if not candidates:
            text = _FALLBACK_NOT_FOUND_EN if language == "en" else _FALLBACK_NOT_FOUND_AR
            return ResponsePacket(text=text, language=language, session_id=session_id)

        candidate_list = ", ".join(f'"{c}"' for c in candidates)
        if language == "ar-EG":
            text = f"لقيت أكتر من نتيجة: {candidate_list}. أي واحد منهم قصدك؟"
        else:
            text = f"I found a few matches: {candidate_list}. Which one did you mean?"

        logger.info("composer_clarification", candidates=candidates, language=language)
        return ResponsePacket(text=text, language=language, session_id=session_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_facts_dict(retrieval: RetrievalResult, original_query: str) -> dict:
        """Convert a RetrievalResult into the facts JSON for the campus prompt."""
        facts: dict = {
            "entity_type": retrieval.entity_type,
            "canonical_name": retrieval.canonical_name,
            "original_query": original_query,
        }
        if retrieval.spoken_facts:
            sf = retrieval.spoken_facts
            facts["building"]      = sf.building
            facts["floor"]         = sf.floor
            facts["room"]          = sf.room
            facts["description"]   = sf.description
            facts["office_hours"]  = sf.office_hours
            facts["contact_notes"] = sf.contact_notes
        facts["nav_code"] = retrieval.nav_code
        facts["nav_safety_notes"] = retrieval.nav_safety_notes
        return facts

    @staticmethod
    def _clean_spoken(raw: str) -> str:
        """Strip JSON wrappers or markdown if the model accidentally included them."""
        text = raw.strip()
        # If model wrapped in JSON, try to extract a 'text' or 'response' key
        if text.startswith("{"):
            try:
                data = json.loads(text)
                for key in ("text", "response", "answer", "reply", "message"):
                    if key in data:
                        return str(data[key]).strip()
            except json.JSONDecodeError:
                pass
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return text

    @staticmethod
    def _fallback_from_facts(retrieval: RetrievalResult, language: str) -> str:
        """Build a minimal spoken answer from raw facts without LLM."""
        name = retrieval.canonical_name or "that location"
        sf = retrieval.spoken_facts

        parts = []
        if sf:
            if sf.building:
                parts.append(f"in {sf.building}")
            if sf.floor:
                parts.append(f"floor {sf.floor}")
            if sf.room:
                parts.append(f"room {sf.room}")

        location_str = ", ".join(parts)
        if language == "ar-EG":
            return f"{name} موجود {location_str}." if location_str else f"لقيت {name} في قاعدة البيانات."
        return f"{name} is {location_str}." if location_str else f"I found {name} in the database."
