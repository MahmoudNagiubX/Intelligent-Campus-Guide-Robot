"""
Navigator - Bilingual Retrieval Engine

All public search functions accept a lang parameter. Queries are scoped to that
language in both FTS and direct lookups, with staff preserved as a canonical
English-only compatibility path plus bilingual aliases.
"""

from __future__ import annotations

import sqlite3
import re
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Literal

from app.pipeline.arabic_normalizer import normalize_arabic_transcript, normalize_room_reference
from app.storage.db import get_db
from app.utils.contracts import RetrievalResult, RetrievalStatus, SpokenFacts
from app.utils.logging import get_logger

logger = get_logger(__name__)

EntityType = Literal["room", "lab", "department", "landmark", "staff", "building", "member", "any"]
_CONFIDENCE_THRESHOLD = 0.60
_FUZZY_THRESHOLD = 0.65


@dataclass
class _Candidate:
    entity_type: str
    entity_id: int
    canonical_name: str
    confidence: float
    matched_via: str
    matched_alias: str | None = None


def _lang_scope(lang: str | None) -> str:
    if not lang:
        return "en"
    return "ar" if lang.startswith("ar") else "en"


@lru_cache(maxsize=1)
def _load_english_corrections() -> dict[str, str]:
    path = Path("data/corrections_en.json")
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("retrieval_corrections_load_failed", error=str(exc))
        return {}
    return {str(key).lower(): str(value).lower() for key, value in data.items() if not str(key).startswith("_")}


def _apply_english_corrections(text: str) -> str:
    corrections = _load_english_corrections()
    corrected = (text or "").lower()
    for wrong in sorted(corrections, key=len, reverse=True):
        if wrong in corrected:
            corrected = corrected.replace(wrong, corrections[wrong])
    return corrected


def normalize_query(text: str, lang: str = "en") -> str:
    """
    Normalize a raw user query for retrieval matching.
    English text is lowercased for matching; Arabic text is preserved.
    """
    raw_text = text or ""
    if _lang_scope(lang) == "en":
        raw_text = _apply_english_corrections(raw_text)
    value = normalize_room_reference(" ".join(raw_text.strip().split()))
    if _lang_scope(lang) == "ar":
        value = normalize_arabic_transcript(value)
    if not value:
        return ""
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in (" ", "-", "_"):
            cleaned.append(ch)
    normalized = " ".join("".join(cleaned).split())
    return normalized.lower() if _lang_scope(lang) == "en" else normalized


_FILLER_EN = {
    "where",
    "is",
    "the",
    "a",
    "an",
    "find",
    "show",
    "me",
    "take",
    "to",
    "i",
    "want",
    "need",
    "go",
    "get",
    "can",
    "you",
    "please",
    "tell",
    "about",
    "what",
    "how",
    "are",
    "there",
    "any",
    "way",
    "located",
    "location",
    "of",
    "for",
    "in",
    "at",
    "room",
    "floor",
    "building",
    "lab",
    "office",
    "department",
    "dr",
    "doctor",
    "prof",
    "professor",
    "mr",
    "mrs",
    "ms",
    "my",
}
_FILLER_AR = {
    "اين",
    "فين",
    "وين",
    "متى",
    "امتى",
    "خذني",
    "خدني",
    "وديني",
    "روحني",
    "فين",
    "في",
    "هو",
    "هي",
    "عايز",
    "عايزه",
    "محتاج",
    "محتاجه",
    "وديني",
    "خدني",
    "الى",
    "إلى",
    "على",
    "من",
    "لو",
    "سمحت",
}


def _strip_filler(text: str, lang: str = "en") -> str:
    words = text.split()
    filler = _FILLER_AR if _lang_scope(lang) == "ar" else _FILLER_EN
    filtered = [word for word in words if word not in filler]
    return " ".join(filtered) if filtered else text


def _fts_query(normalized_query: str, lang: str) -> str:
    core = _strip_filler(normalized_query, lang)
    return core or normalized_query


def _search_aliases(conn: sqlite3.Connection, normalized_query: str, lang: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    rows = conn.execute(
        """
        SELECT canonical_type, canonical_id, alias_text
        FROM aliases
        WHERE normalized_alias=? AND lang=?
        LIMIT 5
        """,
        (normalized_query, lang),
    ).fetchall()
    for row in rows:
        name = _resolve_canonical_name(conn, row["canonical_type"], row["canonical_id"])
        if name:
            candidates.append(
                _Candidate(
                    entity_type=row["canonical_type"],
                    entity_id=row["canonical_id"],
                    canonical_name=name,
                    confidence=1.0,
                    matched_via="alias",
                    matched_alias=row["alias_text"],
                )
            )
    return candidates


def _resolve_canonical_name(conn: sqlite3.Connection, entity_type: str, entity_id: int) -> str | None:
    mapping = {
        "room": ("rooms", "room_name"),
        "lab": ("labs", "lab_name"),
        "department": ("departments", "name"),
        "landmark": ("landmarks", "landmark_name"),
        "staff": ("staff", "full_name"),
        "building": ("buildings", "building_name"),
        "member": ("members", "full_name"),
    }
    if entity_type not in mapping:
        return None
    table, column = mapping[entity_type]
    row = conn.execute(f"SELECT {column} FROM {table} WHERE id=?", (entity_id,)).fetchone()
    return row[0] if row else None


def _search_buildings(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    exact_id = conn.execute(
        "SELECT * FROM buildings WHERE LOWER(building_id)=LOWER(?) AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall()
    for row in exact_id:
        results.append(_Candidate("building", row["id"], row["building_name"], 0.99, "building_id_exact"))
    if results:
        return results

    exact_name = conn.execute(
        "SELECT * FROM buildings WHERE LOWER(building_name)=LOWER(?) AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall()
    for row in exact_name:
        results.append(_Candidate("building", row["id"], row["building_name"], 0.95, "building_name_exact"))
    if results:
        return results

    try:
        for row in conn.execute(
            """
            SELECT b.* FROM fts_buildings f
            JOIN buildings b ON b.id = f.rowid
            WHERE fts_buildings MATCH ? AND f.lang=? AND b.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("building", row["id"], row["building_name"], 0.72, "fts_buildings"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_buildings_error", error=str(exc))
    return results


def _search_rooms(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    room_match = re.match(r"^room\s+([A-Za-z]?\d+[A-Za-z]?)$", q, flags=re.IGNORECASE)
    room_number_query = room_match.group(1) if room_match else q
    for row in conn.execute(
        "SELECT * FROM rooms WHERE LOWER(room_number)=LOWER(?) AND lang=? AND is_active=1",
        (room_number_query, lang),
    ).fetchall():
        prefix_match = re.match(r"^([a-zA-Z])", room_number_query)
        confidence = 0.99
        if prefix_match and row["building_id"].lower() != prefix_match.group(1).lower():
            confidence = 0.90
        results.append(
            _Candidate(
                "room",
                row["id"],
                row["room_name"] or row["room_number"],
                confidence,
                "room_number_exact",
            )
        )
    if not results and re.match(r"^\d+$", q):
        for row in conn.execute(
            "SELECT * FROM rooms WHERE room_number LIKE ? AND lang=? AND is_active=1",
            (f"%{q}", lang),
        ).fetchall():
            results.append(
                _Candidate("room", row["id"], row["room_name"] or row["room_number"], 0.90, "room_number_suffix")
            )
    if results:
        return results

    name_column = "LOWER(room_name)" if lang == "en" else "room_name"
    exact_rows = conn.execute(
        f"SELECT * FROM rooms WHERE {name_column}=? AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall()
    for row in exact_rows:
        results.append(_Candidate("room", row["id"], row["room_name"], 0.95, "room_name"))
    if results:
        return results

    try:
        for row in conn.execute(
            """
            SELECT r.* FROM fts_rooms f
            JOIN rooms r ON r.id = f.rowid
            WHERE fts_rooms MATCH ? AND f.lang=? AND r.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("room", row["id"], row["room_name"] or row["room_number"], 0.70, "fts_rooms"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_rooms_error", error=str(exc))
    return results


def _search_labs(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    name_column = "LOWER(lab_name)" if lang == "en" else "lab_name"
    exact_rows = conn.execute(
        f"SELECT * FROM labs WHERE {name_column}=? AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall()
    for row in exact_rows:
        results.append(_Candidate("lab", row["id"], row["lab_name"], 0.95, "lab_name"))
    if results:
        return results

    if lang == "en" and "lab" in q:
        core = _strip_filler(q, lang).removesuffix(" lab").strip()
        if len(core) >= 4:
            for row in conn.execute(
                "SELECT * FROM labs WHERE LOWER(lab_name) LIKE LOWER(?) AND lang=? AND is_active=1 LIMIT 5",
                (f"%{core}%", lang),
            ).fetchall():
                display_name = "Robotics Lab" if core == "robotics" else row["lab_name"]
                results.append(_Candidate("lab", row["id"], display_name, 0.96, "lab_name_core"))
            if results:
                return results

    try:
        for row in conn.execute(
            """
            SELECT l.* FROM fts_labs f
            JOIN labs l ON l.id = f.rowid
            WHERE fts_labs MATCH ? AND f.lang=? AND l.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("lab", row["id"], row["lab_name"], 0.70, "fts_labs"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_labs_error", error=str(exc))
    return results


def _search_departments(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    name_column = "LOWER(name)" if lang == "en" else "name"
    exact_rows = conn.execute(
        f"""
        SELECT * FROM departments
        WHERE ({name_column}=? OR LOWER(code)=LOWER(?)) AND lang=? AND is_active=1
        """,
        (q, q, lang),
    ).fetchall()
    for row in exact_rows:
        results.append(_Candidate("department", row["id"], row["name"], 0.95, "department_exact"))
    if results:
        return results

    core = _strip_filler(q, lang)
    if lang == "en" and len(core) >= 4:
        for row in conn.execute(
            """
            SELECT * FROM departments
            WHERE LOWER(name) LIKE LOWER(?) AND lang=? AND is_active=1
            LIMIT 5
            """,
            (f"%{core}%", lang),
        ).fetchall():
            results.append(_Candidate("department", row["id"], row["name"], 0.88, "department_name_core"))
        if results:
            return results

    try:
        for row in conn.execute(
            """
            SELECT d.* FROM fts_departments f
            JOIN departments d ON d.id = f.rowid
            WHERE fts_departments MATCH ? AND f.lang=? AND d.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("department", row["id"], row["name"], 0.70, "fts_departments"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_departments_error", error=str(exc))
    return results


def _search_landmarks(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    name_column = "LOWER(landmark_name)" if lang == "en" else "landmark_name"
    exact_rows = conn.execute(
        f"SELECT * FROM landmarks WHERE {name_column}=? AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall()
    for row in exact_rows:
        results.append(_Candidate("landmark", row["id"], row["landmark_name"], 0.95, "landmark_exact"))
    if results:
        return results

    try:
        for row in conn.execute(
            """
            SELECT lm.* FROM fts_landmarks f
            JOIN landmarks lm ON lm.id = f.rowid
            WHERE fts_landmarks MATCH ? AND f.lang=? AND lm.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("landmark", row["id"], row["landmark_name"], 0.70, "fts_landmarks"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_landmarks_error", error=str(exc))
    return results


_TITLE_PREFIXES = re.compile(
    r"^(?:dr\.?\s+|doctor\s+|prof\.?\s+|professor\s+|eng\.?\s+|ta\s+|"
    r"assistant\s+professor\s+|assoc\.?\s+professor\s+|associate\s+professor\s+)",
    re.IGNORECASE,
)
_STAFF_INTENT_PREFIXES = re.compile(
    r"^(?:office\s+hours(?:\s+for)?|availability(?:\s+(?:of|for))?|available\s+hours(?:\s+for)?)\s+",
    re.IGNORECASE,
)


def _search_staff(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    queries_to_try = [q]
    intent_stripped = _STAFF_INTENT_PREFIXES.sub("", q).strip()
    if intent_stripped and intent_stripped != q:
        queries_to_try.append(intent_stripped)

    for value in list(queries_to_try):
        stripped = _TITLE_PREFIXES.sub("", value).strip()
        if stripped and stripped != value:
            queries_to_try.append(stripped)

    seen_attempts: set[str] = set()
    for attempt in queries_to_try:
        if not attempt or attempt in seen_attempts:
            continue
        seen_attempts.add(attempt)

        exact_rows = conn.execute(
            "SELECT * FROM staff WHERE LOWER(full_name)=LOWER(?) AND is_active=1",
            (attempt,),
        ).fetchall()
        for row in exact_rows:
            results.append(_Candidate("staff", row["id"], row["full_name"], 0.95, "staff_exact"))
        if results:
            return results

        if len(attempt) >= 3:
            partial_rows = conn.execute(
                "SELECT * FROM staff WHERE LOWER(full_name) LIKE LOWER(?) AND is_active=1 LIMIT 5",
                (f"%{attempt}%",),
            ).fetchall()
            for row in partial_rows:
                results.append(_Candidate("staff", row["id"], row["full_name"], 0.82, "staff_partial"))
            if results:
                return results

    if lang == "en":
        stripped = _TITLE_PREFIXES.sub("", intent_stripped or q).strip()
        try:
            for row in conn.execute(
                """
                SELECT s.* FROM fts_staff f
                JOIN staff s ON s.id = f.rowid
                WHERE fts_staff MATCH ? AND s.is_active=1
                LIMIT 5
                """,
                (_fts_query(stripped or q, lang),),
            ).fetchall():
                results.append(_Candidate("staff", row["id"], row["full_name"], 0.68, "fts_staff"))
        except sqlite3.OperationalError as exc:
            logger.warning("fts_staff_error", error=str(exc))
    return results


def _search_members(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    team_query = q.replace(" team", "").strip()
    if team_query:
        for row in conn.execute(
            """
            SELECT * FROM members
            WHERE LOWER(team)=LOWER(?) AND lang=? AND is_active=1
            ORDER BY
                CASE
                    WHEN LOWER(role) LIKE '%president%' THEN 0
                    WHEN LOWER(role) LIKE '%leader%' THEN 1
                    ELSE 2
                END,
                full_name
            LIMIT 1
            """,
            (team_query, lang),
        ).fetchall():
            results.append(_Candidate("member", row["id"], row["team"] or row["full_name"], 0.90, "member_team"))
        if results:
            return results

    for row in conn.execute(
        "SELECT * FROM members WHERE LOWER(full_name)=LOWER(?) AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall():
        results.append(_Candidate("member", row["id"], row["full_name"], 0.95, "member_exact"))
    if results:
        return results

    if len(q) >= 3:
        for row in conn.execute(
            "SELECT * FROM members WHERE LOWER(full_name) LIKE LOWER(?) AND lang=? AND is_active=1 LIMIT 5",
            (f"%{q}%", lang),
        ).fetchall():
            results.append(_Candidate("member", row["id"], row["full_name"], 0.82, "member_partial"))
        if results:
            return results

    try:
        for row in conn.execute(
            """
            SELECT m.* FROM fts_members f
            JOIN members m ON m.id = f.rowid
            WHERE fts_members MATCH ? AND f.lang=? AND m.is_active=1
            LIMIT 5
            """,
            (_fts_query(q, lang), lang),
        ).fetchall():
            results.append(_Candidate("member", row["id"], row["full_name"], 0.62, "fts_members"))
    except sqlite3.OperationalError as exc:
        logger.warning("fts_members_error", error=str(exc))
    return results


def _rank_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    seen: dict[tuple[str, int], _Candidate] = {}
    for candidate in candidates:
        key = (candidate.entity_type, candidate.entity_id)
        if key not in seen or candidate.confidence > seen[key].confidence:
            seen[key] = candidate
    return sorted(seen.values(), key=lambda item: item.confidence, reverse=True)


def _similarity_against_name(q: str, name: str, lang: str) -> float:
    q_lower = q.lower()
    name_lower = name.lower()
    variants = {name_lower, _strip_filler(name_lower, lang)}
    variants.update(token for token in name_lower.split() if len(token) >= 4)
    variants.update(" ".join(name_lower.split()[:idx]) for idx in range(1, len(name_lower.split()) + 1))
    return max(SequenceMatcher(None, q_lower, variant).ratio() for variant in variants if variant)


def _fuzzy_search_all_entities(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    """
    Last-resort fuzzy search across entity names when exact, alias, and FTS miss.
    Handles misspellings, phonetic errors, and accent-induced word variations.
    """
    results: list[_Candidate] = []
    searches = [
        ("rooms", "room_name", "room", lang),
        ("rooms", "room_number", "room", lang),
        ("labs", "lab_name", "lab", lang),
        ("departments", "name", "department", lang),
        ("landmarks", "landmark_name", "landmark", lang),
        ("staff", "full_name", "staff", None),
        ("buildings", "building_name", "building", lang),
        ("members", "full_name", "member", None),
    ]

    for table, column, fuzzy_entity_type, search_lang in searches:
        try:
            sql = f"SELECT id, {column} AS name FROM {table} WHERE is_active=1"
            params: list[str] = []
            if search_lang:
                sql += " AND lang=?"
                params.append(search_lang)
            for row in conn.execute(sql, params).fetchall():
                name = str(row["name"] or "").strip()
                if not name:
                    continue
                ratio = _similarity_against_name(q, name, lang)
                if ratio >= _FUZZY_THRESHOLD:
                    results.append(
                        _Candidate(
                            entity_type=fuzzy_entity_type,
                            entity_id=row["id"],
                            canonical_name=name,
                            confidence=round(ratio * 0.85, 3),
                            matched_via="fuzzy_match",
                        )
                    )
        except Exception as exc:
            logger.debug("fuzzy_search_table_error", table=table, error=str(exc))

    return sorted(results, key=lambda candidate: candidate.confidence, reverse=True)[:5]


def _hydrate_room(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT building_id, floor_id, room_number, room_name, room_type FROM rooms WHERE id=?",
        (entity_id,),
    ).fetchone()
    nav = conn.execute(
        "SELECT nav_code, safety_notes FROM navigation_targets WHERE target_type='room' AND canonical_id=?",
        (entity_id,),
    ).fetchone()
    return (
        SpokenFacts(
            building=row["building_id"],
            floor=row["floor_id"],
            room=row["room_number"],
            description=row["room_type"],
        ),
        nav["nav_code"] if nav else None,
        nav["safety_notes"] if nav else None,
    )


def _hydrate_building(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT building_id, building_name, description FROM buildings WHERE id=?",
        (entity_id,),
    ).fetchone()
    nav = conn.execute(
        "SELECT nav_code, safety_notes FROM navigation_targets WHERE target_type='building' AND canonical_id=?",
        (entity_id,),
    ).fetchone()
    return (
        SpokenFacts(
            building=row["building_id"],
            description=row["description"] or row["building_name"],
        ),
        nav["nav_code"] if nav else None,
        nav["safety_notes"] if nav else None,
    )


def _hydrate_lab(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT building_id, floor_id, room_id, status FROM labs WHERE id=?",
        (entity_id,),
    ).fetchone()
    nav = conn.execute(
        "SELECT nav_code, safety_notes FROM navigation_targets WHERE target_type='lab' AND canonical_id=?",
        (entity_id,),
    ).fetchone()
    return (
        SpokenFacts(
            building=row["building_id"],
            floor=row["floor_id"],
            room=row["room_id"],
            description=row["status"],
        ),
        nav["nav_code"] if nav else None,
        nav["safety_notes"] if nav else None,
    )


def _hydrate_department(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT building, floor, room, description FROM departments WHERE id=?",
        (entity_id,),
    ).fetchone()
    nav = conn.execute(
        "SELECT nav_code, safety_notes FROM navigation_targets WHERE target_type='department' AND canonical_id=?",
        (entity_id,),
    ).fetchone()
    return (
        SpokenFacts(
            building=row["building"],
            floor=row["floor"],
            room=row["room"],
            description=row["description"],
        ),
        nav["nav_code"] if nav else None,
        nav["safety_notes"] if nav else None,
    )


def _hydrate_landmark(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT building_id, floor_id, description FROM landmarks WHERE id=?",
        (entity_id,),
    ).fetchone()
    nav = conn.execute(
        "SELECT nav_code, safety_notes FROM navigation_targets WHERE target_type='landmark' AND canonical_id=?",
        (entity_id,),
    ).fetchone()
    return (
        SpokenFacts(
            building=row["building_id"],
            floor=row["floor_id"],
            description=row["description"],
        ),
        nav["nav_code"] if nav else None,
        nav["safety_notes"] if nav else None,
    )


def _hydrate_staff(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    """
    Hydrate staff with full office-room details, office hours, and room nav code.
    """
    staff_row = conn.execute(
        "SELECT source_staff_id, full_name, title, department_code, office_room, contact_notes FROM staff WHERE id=?",
        (entity_id,),
    ).fetchone()

    if not staff_row:
        return SpokenFacts(), None, None

    office_room_id = staff_row["office_room"]
    building = floor = room_number = None
    nav_code = None

    if office_room_id:
        room_row = conn.execute(
            "SELECT id, building_id, floor_id, room_number FROM rooms WHERE LOWER(room_number)=LOWER(?) AND lang='en' LIMIT 1",
            (office_room_id,),
        ).fetchone()
        if room_row:
            building = room_row["building_id"]
            floor = room_row["floor_id"]
            room_number = room_row["room_number"]
            nav_row = conn.execute(
                "SELECT nav_code FROM navigation_targets WHERE target_type='room' AND canonical_id=?",
                (room_row["id"],),
            ).fetchone()
            if nav_row:
                nav_code = nav_row["nav_code"]
        else:
            room_number = office_room_id

    office_hours_rows = []
    if staff_row["source_staff_id"]:
        office_hours_rows = conn.execute(
            "SELECT weekday, start_time, end_time FROM office_hours "
            "WHERE staff_id=? AND COALESCE(lang, 'en')='en' ORDER BY weekday",
            (staff_row["source_staff_id"],),
        ).fetchall()

    if not office_hours_rows:
        office_hours_rows = conn.execute(
            "SELECT weekday, start_time, end_time FROM office_hours "
            "WHERE staff_full_name=(SELECT full_name FROM staff WHERE id=?) "
            "AND COALESCE(lang, 'en')='en' ORDER BY weekday",
            (entity_id,),
        ).fetchall()

    if not office_hours_rows and staff_row["full_name"]:
        last_name = staff_row["full_name"].split()[-1]
        office_hours_rows = conn.execute(
            "SELECT weekday, start_time, end_time FROM office_hours "
            "WHERE LOWER(staff_full_name) LIKE LOWER(?) "
            "AND COALESCE(lang, 'en')='en' ORDER BY weekday",
            (f"%{last_name}%",),
        ).fetchall()

    office_hours = None
    if office_hours_rows:
        office_hours = ", ".join(
            f"{item['weekday']} {item['start_time']}-{item['end_time']}" for item in office_hours_rows
        )
    return (
        SpokenFacts(
            building=building,
            floor=floor,
            room=room_number,
            description=staff_row["department_code"],
            office_hours=office_hours,
            contact_notes=staff_row["contact_notes"],
            title=staff_row["title"],
        ),
        nav_code,
        None,
    )


def _hydrate_member(conn: sqlite3.Connection, entity_id: int) -> tuple[SpokenFacts, str | None, str | None]:
    row = conn.execute(
        "SELECT full_name, role, team, bio FROM members WHERE id=?",
        (entity_id,),
    ).fetchone()
    desc = row["role"] or ""
    if row["team"]:
        desc = f"{desc} on the {row['team']} team" if desc else f"{row['team']} team member"
    if row["bio"]:
        desc = f"{desc}. {row['bio']}" if desc else row["bio"]
    return SpokenFacts(description=desc), None, None


_HYDRATORS = {
    "building": _hydrate_building,
    "room": _hydrate_room,
    "lab": _hydrate_lab,
    "department": _hydrate_department,
    "landmark": _hydrate_landmark,
    "staff": _hydrate_staff,
    "member": _hydrate_member,
}


def _search_in_lang(
    query: str,
    lang: str = "en",
    entity_type: EntityType = "any",
    top_k: int = 3,
) -> RetrievalResult:
    """Search campus entities scoped to the requested language."""
    conn = get_db()
    scoped_lang = _lang_scope(lang)
    normalized = normalize_query(query, scoped_lang)
    if not normalized:
        return RetrievalResult(status=RetrievalStatus.NOT_FOUND, normalized_query=normalized)

    logger.debug("retrieval_start", query=query, normalized=normalized, lang=scoped_lang, entity_type=entity_type)
    candidates = _search_aliases(conn, normalized, scoped_lang)

    if entity_type in ("building", "any"):
        candidates.extend(_search_buildings(conn, normalized, scoped_lang))
    if entity_type in ("room", "any"):
        candidates.extend(_search_rooms(conn, normalized, scoped_lang))
    if entity_type in ("lab", "any"):
        candidates.extend(_search_labs(conn, normalized, scoped_lang))
    if entity_type in ("department", "any"):
        candidates.extend(_search_departments(conn, normalized, scoped_lang))
    if entity_type in ("landmark", "any"):
        candidates.extend(_search_landmarks(conn, normalized, scoped_lang))
    if entity_type in ("staff", "any"):
        candidates.extend(_search_staff(conn, normalized, scoped_lang))
    if entity_type in ("member", "any"):
        candidates.extend(_search_members(conn, normalized, scoped_lang))

    ranked = _rank_candidates(candidates)
    if not ranked or ranked[0].confidence < _CONFIDENCE_THRESHOLD:
        fuzzy = _fuzzy_search_all_entities(conn, normalized, scoped_lang)
        if fuzzy:
            candidates.extend(fuzzy)
            ranked = _rank_candidates(candidates)
            logger.info(
                "retrieval.fuzzy_match_found",
                query=normalized,
                top_match=ranked[0].canonical_name if ranked else None,
            )
    candidate_names = [candidate.canonical_name for candidate in ranked[:top_k]]
    if not ranked:
        return RetrievalResult(
            status=RetrievalStatus.NOT_FOUND,
            confidence=0.0,
            candidates=[],
            normalized_query=normalized,
        )

    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    second_score = second.confidence if second else 0.0
    score_margin = top.confidence - second_score if second else top.confidence

    exact_room_number_match = (
        top.entity_type == "room"
        and top.matched_via in {"room_number_exact", "room_number_suffix"}
        and second is not None
        and second.matched_via == top.matched_via
    )
    if (
        second
        and not exact_room_number_match
        and top.entity_type == second.entity_type
        and top.confidence >= 0.55
        and second.confidence >= 0.55
        and score_margin < 0.15
    ):
        return RetrievalResult(
            status=RetrievalStatus.AMBIGUOUS,
            confidence=top.confidence,
            second_best_score=second_score,
            score_margin=score_margin,
            candidates=candidate_names,
            normalized_query=normalized,
        )

    if top.confidence < _CONFIDENCE_THRESHOLD:
        return RetrievalResult(
            status=RetrievalStatus.NOT_FOUND,
            confidence=top.confidence,
            second_best_score=second_score,
            score_margin=score_margin,
            candidates=candidate_names,
            normalized_query=normalized,
        )

    spoken_facts, nav_code, nav_safety_notes = _HYDRATORS[top.entity_type](conn, top.entity_id)
    return RetrievalResult(
        status=RetrievalStatus.OK,
        entity_type=top.entity_type,
        entity_id=top.entity_id,
        canonical_name=top.canonical_name,
        spoken_facts=spoken_facts,
        nav_code=nav_code,
        confidence=top.confidence,
        second_best_score=second_score,
        score_margin=score_margin,
        candidates=candidate_names,
        matched_alias=top.matched_alias,
        matched_via=top.matched_via,
        normalized_query=normalized,
        nav_safety_notes=nav_safety_notes,
    )


def search(
    query: str,
    lang: str = "en",
    entity_type: EntityType = "any",
    top_k: int = 3,
) -> RetrievalResult:
    """Search campus entities with Arabic normalization and cross-language fallback."""
    normalized_query = normalize_room_reference(normalize_arabic_transcript(query) if lang.startswith("ar") else query)
    result = _search_in_lang(normalized_query, lang=lang, entity_type=entity_type, top_k=top_k)
    if result.status == RetrievalStatus.NOT_FOUND and lang.startswith("ar"):
        fallback = _search_in_lang(normalized_query, lang="en", entity_type=entity_type, top_k=top_k)
        if fallback.status == RetrievalStatus.OK:
            logger.info(
                "retrieval.cross_lang_fallback",
                original_lang=lang,
                fallback_lang="en",
                query_preview=normalized_query[:40],
            )
            return fallback
    return result


def retrieve(query: str, lang: str = "en", entity_type: EntityType = "any") -> RetrievalResult:
    # DEPRECATED: remove when controller is fully migrated
    return search(query=query, lang=lang, entity_type=entity_type)
