"""
Navigator - Bilingual Retrieval Engine

All public search functions accept a lang parameter. Queries are scoped to that
language in both FTS and direct lookups, with staff preserved as a canonical
English-only compatibility path plus bilingual aliases.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from app.storage.db import get_db
from app.utils.contracts import RetrievalResult, RetrievalStatus, SpokenFacts
from app.utils.logging import get_logger

logger = get_logger(__name__)

EntityType = Literal["room", "lab", "department", "landmark", "staff", "any"]
_CONFIDENCE_THRESHOLD = 0.60


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


def normalize_query(text: str, lang: str = "en") -> str:
    """
    Normalize a raw user query for retrieval matching.
    English text is lowercased for matching; Arabic text is preserved.
    """
    value = " ".join((text or "").strip().split())
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
    }
    if entity_type not in mapping:
        return None
    table, column = mapping[entity_type]
    row = conn.execute(f"SELECT {column} FROM {table} WHERE id=?", (entity_id,)).fetchone()
    return row[0] if row else None


def _search_rooms(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    for row in conn.execute(
        "SELECT * FROM rooms WHERE LOWER(room_number)=LOWER(?) AND lang=? AND is_active=1",
        (q, lang),
    ).fetchall():
        results.append(_Candidate("room", row["id"], row["room_name"], 1.0, "room_number"))
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
            results.append(_Candidate("room", row["id"], row["room_name"], 0.70, "fts_rooms"))
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


def _search_staff(conn: sqlite3.Connection, q: str, lang: str) -> list[_Candidate]:
    results: list[_Candidate] = []
    exact_rows = conn.execute(
        "SELECT * FROM staff WHERE LOWER(full_name)=LOWER(?) AND is_active=1",
        (q,),
    ).fetchall()
    for row in exact_rows:
        results.append(_Candidate("staff", row["id"], row["full_name"], 0.95, "staff_exact"))
    if results:
        return results

    if lang == "en":
        try:
            for row in conn.execute(
                """
                SELECT s.* FROM fts_staff f
                JOIN staff s ON s.id = f.rowid
                WHERE fts_staff MATCH ? AND s.is_active=1
                LIMIT 5
                """,
                (_fts_query(q, lang),),
            ).fetchall():
                results.append(_Candidate("staff", row["id"], row["full_name"], 0.68, "fts_staff"))
        except sqlite3.OperationalError as exc:
            logger.warning("fts_staff_error", error=str(exc))
    return results


def _rank_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    seen: dict[tuple[str, int], _Candidate] = {}
    for candidate in candidates:
        key = (candidate.entity_type, candidate.entity_id)
        if key not in seen or candidate.confidence > seen[key].confidence:
            seen[key] = candidate
    return sorted(seen.values(), key=lambda item: item.confidence, reverse=True)


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
    row = conn.execute(
        "SELECT department_code, office_room, contact_notes FROM staff WHERE id=?",
        (entity_id,),
    ).fetchone()
    office_hours_rows = conn.execute(
        "SELECT weekday, start_time, end_time FROM office_hours WHERE staff_full_name=(SELECT full_name FROM staff WHERE id=?) ORDER BY weekday",
        (entity_id,),
    ).fetchall()
    office_hours = None
    if office_hours_rows:
        office_hours = ", ".join(
            f"{item['weekday']} {item['start_time']}-{item['end_time']}" for item in office_hours_rows
        )
    return (
        SpokenFacts(
            building=None,
            floor=None,
            room=row["office_room"],
            description=row["department_code"],
            office_hours=office_hours,
            contact_notes=row["contact_notes"],
        ),
        None,
        None,
    )


_HYDRATORS = {
    "room": _hydrate_room,
    "lab": _hydrate_lab,
    "department": _hydrate_department,
    "landmark": _hydrate_landmark,
    "staff": _hydrate_staff,
}


def search(
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

    ranked = _rank_candidates(candidates)
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

    if second and top.entity_type == second.entity_type and top.confidence >= 0.55 and second.confidence >= 0.55 and score_margin < 0.15:
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


def retrieve(query: str, lang: str = "en", entity_type: EntityType = "any") -> RetrievalResult:
    # DEPRECATED: remove when controller is fully migrated
    return search(query=query, lang=lang, entity_type=entity_type)
