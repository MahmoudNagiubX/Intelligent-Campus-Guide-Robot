"""
Navigator - Retrieval Engine
Turns a raw user query into a verified campus entity from SQLite.

Search order:
1. Normalize the query
2. Exact alias lookup  (fastest, most precise)
3. FTS5 trigram search across all entity tables
4. Rank candidates and apply confidence threshold
5. Return ONE result, an ambiguity set, or not_found

Rules:
- Never invent campus facts.
- Never return a result below the confidence threshold.
- Ambiguity is always preferred over a wrong guess.
"""

import sqlite3
from dataclasses import dataclass

from app.storage.db import get_db
from app.utils.contracts import RetrievalResult, RetrievalStatus, SpokenFacts
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum score to return a single confident result
_CONFIDENCE_THRESHOLD = 0.60

# FTS rank score below this is considered noise
_FTS_MIN_SCORE = -5.0


# ─────────────────────────────────────────────────────────────────────────────
# Query Normalization
# ─────────────────────────────────────────────────────────────────────────────

# Filler words that should be stripped before searching
_FILLER = {
    "where", "is", "the", "a", "an", "find", "show", "me", "take",
    "to", "i", "want", "need", "go", "get", "can", "you", "please",
    "tell", "about", "what", "how", "are", "there", "any", "way",
    "located", "location", "of", "for", "in", "at", "room", "floor",
    "building", "lab", "office", "department", "dr", "doctor", "prof",
    "professor", "mr", "mrs", "ms", "my",
}

_ROOM_PATTERNS = [
    ("lab ", ""),
    ("room ", ""),
    ("floor ", ""),
    ("building ", ""),
]


def normalize_query(text: str) -> str:
    """
    Normalize a raw user query for retrieval matching.
    - Lowercase
    - Collapse whitespace
    - Strip filler words
    - Standardize room/lab/floor markers
    """
    text = text.lower().strip()
    text = " ".join(text.split())

    # Remove punctuation except digits and Arabic chars
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in (" ", "-", "_") or "\u0600" <= ch <= "\u06ff":
            cleaned.append(ch)
    text = "".join(cleaned)

    # Collapse again after punctuation removal
    text = " ".join(text.split())
    return text


def _strip_filler(text: str) -> str:
    """Remove filler words to get the core search term."""
    words = [w for w in text.split() if w not in _FILLER]
    return " ".join(words) if words else text


# ─────────────────────────────────────────────────────────────────────────────
# Candidate dataclass (internal)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    entity_type: str
    entity_id: int
    canonical_name: str
    score: float
    matched_via: str   # 'alias', 'fts_locations', 'fts_staff', etc.


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Alias Lookup
# ─────────────────────────────────────────────────────────────────────────────

def _search_aliases(conn: sqlite3.Connection, normalized_query: str) -> list[_Candidate]:
    """
    Exact match against the aliases table.
    This is the fastest and highest-confidence path.
    """
    candidates: list[_Candidate] = []

    # Try full query first, then core term
    terms = [normalized_query, _strip_filler(normalized_query)]
    terms = list(dict.fromkeys(t for t in terms if t))  # deduplicate, preserve order

    for term in terms:
        rows = conn.execute(
            """SELECT a.canonical_type, a.canonical_id, a.alias_text
               FROM aliases a
               WHERE a.normalized_alias = ?
               LIMIT 5;""",
            (term,),
        ).fetchall()

        for row in rows:
            name = _resolve_canonical_name(conn, row["canonical_type"], row["canonical_id"])
            if name:
                candidates.append(_Candidate(
                    entity_type=row["canonical_type"],
                    entity_id=row["canonical_id"],
                    canonical_name=name,
                    score=1.0,
                    matched_via="alias",
                ))

    return candidates


def _resolve_canonical_name(conn: sqlite3.Connection, entity_type: str, entity_id: int) -> str | None:
    """Look up the canonical name for any entity type."""
    table_map = {
        "location":   ("locations",   "name"),
        "staff":      ("staff",       "full_name"),
        "department": ("departments", "name"),
        "facility":   ("facilities",  "name"),
    }
    if entity_type not in table_map:
        return None
    table, col = table_map[entity_type]
    row = conn.execute(f"SELECT {col} FROM {table} WHERE id=?;", (entity_id,)).fetchone()
    return row[0] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FTS5 Search
# ─────────────────────────────────────────────────────────────────────────────

def _search_fts(conn: sqlite3.Connection, normalized_query: str) -> list[_Candidate]:
    """
    FTS5 trigram search across all entity tables.
    Returns ranked candidates sorted by relevance score.
    """
    candidates: list[_Candidate] = []
    core_term = _strip_filler(normalized_query)
    search_term = core_term if core_term else normalized_query

    fts_queries = [
        # (fts_table, entity_type, id_col, name_col)
        ("fts_locations",   "location",   "locations",   "name"),
        ("fts_staff",       "staff",      "staff",       "full_name"),
        ("fts_departments", "department", "departments", "name"),
        ("fts_facilities",  "facility",   "facilities",  "name"),
    ]

    for fts_table, entity_type, base_table, name_col in fts_queries:
        try:
            rows = conn.execute(
                f"""SELECT b.id, b.{name_col}, bm25({fts_table}) AS score
                    FROM {fts_table}
                    JOIN {base_table} b ON b.id = {fts_table}.rowid
                    WHERE {fts_table} MATCH ?
                    ORDER BY score
                    LIMIT 5;""",
                (search_term,),
            ).fetchall()

            for row in rows:
                raw_score = row["score"]
                if raw_score < _FTS_MIN_SCORE:
                    continue
                # Normalize bm25 score (negative → positive 0..1)
                normalized_score = max(0.0, min(1.0, 1.0 + raw_score / 10.0))
                candidates.append(_Candidate(
                    entity_type=entity_type,
                    entity_id=row["id"],
                    canonical_name=row[name_col],
                    score=normalized_score,
                    matched_via=fts_table,
                ))
        except sqlite3.OperationalError:
            # FTS query syntax error — skip this table
            logger.debug("fts_query_skipped", table=fts_table, term=search_term)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Ranking
# ─────────────────────────────────────────────────────────────────────────────

def _rank_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    """
    Deduplicate by entity and sort by score descending.
    Alias matches always outrank FTS matches.
    """
    seen: dict[tuple[str, int], _Candidate] = {}
    for c in candidates:
        key = (c.entity_type, c.entity_id)
        if key not in seen or c.score > seen[key].score:
            seen[key] = c
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


def _candidate_names(candidates: list[_Candidate], limit: int = 3) -> list[str]:
    """Return the top canonical names for logging and clarification hints."""
    names: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.canonical_name in seen:
            continue
        seen.add(candidate.canonical_name)
        names.append(candidate.canonical_name)
        if len(names) >= limit:
            break
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Entity hydration
# ─────────────────────────────────────────────────────────────────────────────

def _hydrate_location(
    conn: sqlite3.Connection,
    entity_id: int,
) -> tuple[SpokenFacts, str | None, str | None, str | None]:
    """Fetch spoken facts and nav info for a location."""
    row = conn.execute(
        "SELECT building, floor, room, description, map_node FROM locations WHERE id=?;",
        (entity_id,),
    ).fetchone()
    if not row:
        return SpokenFacts(), None, None, None

    facts = SpokenFacts(
        building=row["building"],
        floor=row["floor"],
        room=row["room"],
        description=row["description"],
    )

    nav = conn.execute(
        """SELECT nav_code, safety_notes
           FROM navigation_targets
           WHERE target_type='location' AND canonical_id=?;""",
        (entity_id,),
    ).fetchone()

    return facts, nav["nav_code"] if nav else None, row["map_node"], nav["safety_notes"] if nav else None


def _hydrate_staff(
    conn: sqlite3.Connection,
    entity_id: int,
) -> tuple[SpokenFacts, str | None, str | None, str | None]:
    """Fetch spoken facts for a staff member including office location and hours."""
    row = conn.execute(
        """SELECT s.title, s.contact_notes,
                  l.building, l.floor, l.room
           FROM staff s
           LEFT JOIN locations l ON l.id = s.office_location_id
           WHERE s.id=?;""",
        (entity_id,),
    ).fetchone()
    if not row:
        return SpokenFacts(), None, None, None

    # Aggregate office hours
    hours_rows = conn.execute(
        "SELECT weekday, start_time, end_time FROM office_hours WHERE staff_id=? ORDER BY weekday;",
        (entity_id,),
    ).fetchall()
    hours_text = None
    if hours_rows:
        parts = [f"{r['weekday']} {r['start_time']}–{r['end_time']}" for r in hours_rows]
        hours_text = ", ".join(parts)

    facts = SpokenFacts(
        building=row["building"],
        floor=row["floor"],
        room=row["room"],
        office_hours=hours_text,
        contact_notes=row["contact_notes"],
    )
    return facts, None, None, None


def _hydrate_department(
    conn: sqlite3.Connection,
    entity_id: int,
) -> tuple[SpokenFacts, str | None, str | None, str | None]:
    row = conn.execute(
        "SELECT building, floor, room, description FROM departments WHERE id=?;",
        (entity_id,),
    ).fetchone()
    if not row:
        return SpokenFacts(), None, None, None
    facts = SpokenFacts(
        building=row["building"],
        floor=row["floor"],
        room=row["room"],
        description=row["description"],
    )
    nav = conn.execute(
        """SELECT nav_code, safety_notes
           FROM navigation_targets
           WHERE target_type='department' AND canonical_id=?;""",
        (entity_id,),
    ).fetchone()
    return facts, nav["nav_code"] if nav else None, None, nav["safety_notes"] if nav else None


def _hydrate_facility(
    conn: sqlite3.Connection,
    entity_id: int,
) -> tuple[SpokenFacts, str | None, str | None, str | None]:
    row = conn.execute(
        "SELECT building, floor, room, description FROM facilities WHERE id=?;",
        (entity_id,),
    ).fetchone()
    if not row:
        return SpokenFacts(), None, None, None
    facts = SpokenFacts(
        building=row["building"],
        floor=row["floor"],
        room=row["room"],
        description=row["description"],
    )
    nav = conn.execute(
        """SELECT nav_code, safety_notes
           FROM navigation_targets
           WHERE target_type='facility' AND canonical_id=?;""",
        (entity_id,),
    ).fetchone()
    return facts, nav["nav_code"] if nav else None, None, nav["safety_notes"] if nav else None


_HYDRATORS = {
    "location":   _hydrate_location,
    "staff":      _hydrate_staff,
    "department": _hydrate_department,
    "facility":   _hydrate_facility,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(query: str) -> RetrievalResult:
    """
    Main retrieval entry point.
    Takes a raw user query and returns a verified RetrievalResult.

    Args:
        query: Raw transcript text from the user.

    Returns:
        RetrievalResult with status OK, AMBIGUOUS, or NOT_FOUND.
    """
    conn = get_db()
    normalized = normalize_query(query)
    if not normalized:
        return RetrievalResult(status=RetrievalStatus.NOT_FOUND, confidence=0.0, normalized_query=normalized)

    logger.debug("retrieval_start", query=query, normalized=normalized)

    # Phase 1: alias lookup
    candidates = _search_aliases(conn, normalized)

    # Phase 2: FTS if aliases didn't give a clear answer
    if len(candidates) < 1:
        fts_hits = _search_fts(conn, normalized)
        candidates.extend(fts_hits)

    # Phase 3: rank
    ranked = _rank_candidates(candidates)
    ranked_names = _candidate_names(ranked)

    if not ranked:
        logger.info("retrieval_not_found", query=query)
        return RetrievalResult(
            status=RetrievalStatus.NOT_FOUND,
            confidence=0.0,
            normalized_query=normalized,
        )

    top = ranked[0]
    second = ranked[1] if len(ranked) >= 2 else None
    second_score = second.score if second else 0.0
    score_gap = top.score - second_score if second else top.score

    # Phase 4: ambiguity check
    # Alias matches (score=1.0) are always definitive — skip ambiguity check.
    # For FTS: only flag ambiguity between same entity-type candidates.
    if top.matched_via == "alias":
        pass  # alias wins unconditionally
    elif len(ranked) >= 2:
        same_type = top.entity_type == second.entity_type
        both_strong = top.score >= _CONFIDENCE_THRESHOLD and second.score >= _CONFIDENCE_THRESHOLD
        if same_type and both_strong and score_gap < 0.15:
            logger.info("retrieval_ambiguous", query=query, candidates=ranked_names, score_gap=round(score_gap, 3))
            return RetrievalResult(
                status=RetrievalStatus.AMBIGUOUS,
                candidates=ranked_names,
                confidence=top.score,
                second_best_score=second_score,
                score_margin=score_gap,
                normalized_query=normalized,
            )

    # Phase 5: confidence gate
    if top.score < _CONFIDENCE_THRESHOLD:
        logger.info(
            "retrieval_low_confidence",
            query=query,
            score=top.score,
            top=top.canonical_name,
            second_score=round(second_score, 3),
            score_gap=round(score_gap, 3),
        )
        return RetrievalResult(
            status=RetrievalStatus.NOT_FOUND,
            confidence=top.score,
            candidates=ranked_names,
            second_best_score=second_score,
            score_margin=score_gap,
            normalized_query=normalized,
        )

    # Phase 6: hydrate the winner
    hydrator = _HYDRATORS.get(top.entity_type)
    if not hydrator:
        return RetrievalResult(status=RetrievalStatus.NOT_FOUND)

    spoken_facts, nav_code, map_node, nav_safety_notes = hydrator(conn, top.entity_id)

    logger.info(
        "retrieval_ok",
        query=query,
        entity_type=top.entity_type,
        canonical_name=top.canonical_name,
        score=round(top.score, 3),
        matched_via=top.matched_via,
        second_score=round(second_score, 3),
        score_gap=round(score_gap, 3),
    )

    return RetrievalResult(
        status=RetrievalStatus.OK,
        entity_type=top.entity_type,
        entity_id=top.entity_id,
        canonical_name=top.canonical_name,
        spoken_facts=spoken_facts,
        nav_code=nav_code,
        map_node=map_node,
        confidence=top.score,
        second_best_score=second_score,
        score_margin=score_gap,
        candidates=ranked_names,
        matched_via=top.matched_via,
        normalized_query=normalized,
        nav_safety_notes=nav_safety_notes,
    )
