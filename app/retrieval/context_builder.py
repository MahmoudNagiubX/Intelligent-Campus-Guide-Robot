"""
Build rich, joined context from the SQLite truth layer for a retrieved entity.

The response composer injects this context into the campus prompt so ino can
answer naturally without inventing details.
"""

from __future__ import annotations

from sqlite3 import Connection

from app.storage.db import get_db
from app.utils.contracts import RetrievalResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_rich_context(retrieval: RetrievalResult) -> str:
    """Build comprehensive verified context for a DB retrieval hit."""
    if not retrieval or not retrieval.spoken_facts:
        return ""

    conn = get_db()
    facts = retrieval.spoken_facts
    parts: list[str] = []

    parts.append(f"Name: {retrieval.canonical_name}")
    parts.append(f"Entity: {retrieval.canonical_name}")
    entity_label = str(retrieval.entity_type or "").replace("_", " ").title()
    parts.append(f"Type: {entity_label}")

    try:
        if retrieval.entity_type == "staff":
            parts.extend(_build_staff_context(conn, retrieval))
        elif retrieval.entity_type == "member":
            parts.extend(_build_member_context(conn, retrieval))
        elif retrieval.entity_type == "lab":
            parts.extend(_build_lab_context(conn, retrieval))
        elif retrieval.entity_type == "room":
            parts.extend(_build_room_context(conn, retrieval))
        elif retrieval.entity_type == "department":
            parts.extend(_build_department_context(conn, retrieval))
        elif retrieval.entity_type == "building":
            parts.extend(_build_building_context(conn, retrieval))
        else:
            parts.extend(_basic_fact_lines(retrieval))
    except Exception as exc:
        logger.warning("rich_context_build_failed", entity_type=retrieval.entity_type, error=str(exc))
        parts.extend(_basic_fact_lines(retrieval))

    if retrieval.nav_code:
        parts.append(f"Nav code: {retrieval.nav_code}")
        parts.append(f"Navigation: Available (code {retrieval.nav_code})")
    if facts.office_hours:
        parts.append(f"Office hours: {facts.office_hours}")
    if facts.contact_notes:
        parts.append(f"Contact: {facts.contact_notes}")

    return "\n".join(_dedupe_keep_order(parts))


def _basic_fact_lines(retrieval: RetrievalResult) -> list[str]:
    facts = retrieval.spoken_facts
    if not facts:
        return []
    parts: list[str] = []
    if facts.title:
        parts.append(f"Title: {facts.title}")
    if facts.building:
        parts.append(f"Building: {facts.building}")
    if facts.floor:
        parts.append(f"Floor: {facts.floor}")
    if facts.room:
        parts.append(f"Room: {facts.room}")
    if facts.description:
        parts.append(f"Description: {facts.description}")
    return parts


def _build_staff_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    facts = retrieval.spoken_facts
    if not facts:
        return []

    parts = _basic_fact_lines(retrieval)
    if facts.building and facts.floor and facts.room:
        parts.append(f"Office: Building {facts.building}, Floor {facts.floor}, Room {facts.room}")
    elif facts.room:
        parts.append(f"Office room: {facts.room}")

    if facts.description:
        colleagues = conn.execute(
            """
            SELECT full_name, title
            FROM staff
            WHERE department_code=? AND id!=? AND lang='en' AND is_active=1
            ORDER BY full_name
            LIMIT 3
            """,
            (facts.description, retrieval.entity_id),
        ).fetchall()
        if colleagues:
            names = [f"{row['title']} {row['full_name']}".strip() for row in colleagues]
            parts.append(f"Nearby department colleagues: {', '.join(names)}")

    return parts


def _build_member_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    row = conn.execute(
        "SELECT full_name, role, team, bio FROM members WHERE id=?",
        (retrieval.entity_id,),
    ).fetchone()
    if not row:
        return _basic_fact_lines(retrieval)

    parts = [f"Name: {row['full_name']}"]
    if row["role"]:
        parts.append(f"Role: {row['role']}")
    if row["team"]:
        parts.append(f"Team: {row['team']}")
    if row["bio"]:
        parts.append(f"Bio: {row['bio']}")
    return parts


def _build_lab_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    row = conn.execute(
        "SELECT lab_name, building_id, floor_id, room_id, status FROM labs WHERE id=?",
        (retrieval.entity_id,),
    ).fetchone()
    if not row:
        return _basic_fact_lines(retrieval)

    parts = [
        f"Lab name: {row['lab_name']}",
        f"Building: {row['building_id']}",
    ]
    if row["floor_id"]:
        parts.append(f"Floor: {row['floor_id']}")
    if row["room_id"]:
        parts.append(f"Room: {row['room_id']}")
        room = conn.execute(
            "SELECT room_name, room_type FROM rooms WHERE room_number=? AND lang='en' AND is_active=1 LIMIT 1",
            (row["room_id"],),
        ).fetchone()
        if room:
            parts.append(f"Room record: {room['room_name']} ({room['room_type']})")
    if row["status"]:
        parts.append(f"Status: {row['status']}")
    return parts


def _build_room_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    row = conn.execute(
        "SELECT room_number, room_name, room_type, building_id, floor_id FROM rooms WHERE id=?",
        (retrieval.entity_id,),
    ).fetchone()
    if not row:
        return _basic_fact_lines(retrieval)

    parts = [
        f"Room number: {row['room_number']}",
        f"Room name: {row['room_name']}",
        f"Room type: {row['room_type']}",
        f"Building: {row['building_id']}",
    ]
    if row["floor_id"]:
        parts.append(f"Floor: {row['floor_id']}")

    labs = conn.execute(
        "SELECT lab_name FROM labs WHERE room_id=? AND lang='en' AND is_active=1 ORDER BY lab_name",
        (row["room_number"],),
    ).fetchall()
    if labs:
        parts.append(f"Labs in this room: {', '.join(item['lab_name'] for item in labs)}")

    staff = conn.execute(
        "SELECT full_name, title FROM staff WHERE office_room=? AND lang='en' AND is_active=1 ORDER BY full_name LIMIT 5",
        (row["room_number"],),
    ).fetchall()
    if staff:
        names = [f"{item['title']} {item['full_name']}".strip() for item in staff]
        parts.append(f"Staff offices here: {', '.join(names)}")

    depts = conn.execute(
        """
        SELECT name, code
        FROM departments
        WHERE (room=? OR head_room=?) AND lang='en' AND is_active=1
        ORDER BY name
        """,
        (row["room_number"], row["room_number"]),
    ).fetchall()
    if depts:
        dept_names = [f"{item['name']} ({item['code']})" for item in depts]
        parts.append(f"Departments here: {', '.join(dept_names)}")

    return parts


def _build_department_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    row = conn.execute(
        "SELECT code, name, building, floor, room, head_room, description FROM departments WHERE id=?",
        (retrieval.entity_id,),
    ).fetchone()
    if not row:
        return _basic_fact_lines(retrieval)

    parts = [
        f"Department: {row['name']}",
        f"Code: {row['code']}",
    ]
    if row["building"]:
        parts.append(f"Building: {row['building']}")
    if row["floor"]:
        parts.append(f"Floor: {row['floor']}")
    if row["room"]:
        parts.append(f"Office room: {row['room']}")
    if row["head_room"]:
        parts.append(f"Head office room: {row['head_room']}")
    if row["description"]:
        parts.append(f"Description: {row['description']}")

    staff = conn.execute(
        """
        SELECT full_name, title, office_room
        FROM staff
        WHERE department_code=? AND lang='en' AND is_active=1
        ORDER BY full_name
        LIMIT 6
        """,
        (row["code"],),
    ).fetchall()
    if staff:
        names = [
            f"{item['title']} {item['full_name']} ({item['office_room']})".strip()
            for item in staff
        ]
        parts.append(f"Staff: {', '.join(names)}")

    return parts


def _build_building_context(conn: Connection, retrieval: RetrievalResult) -> list[str]:
    row = conn.execute(
        "SELECT building_id, building_name, description FROM buildings WHERE id=?",
        (retrieval.entity_id,),
    ).fetchone()
    if not row:
        return _basic_fact_lines(retrieval)

    parts = [
        f"Building ID: {row['building_id']}",
        f"Building name: {row['building_name']}",
    ]
    if row["description"]:
        parts.append(f"Description: {row['description']}")

    room_count = conn.execute(
        "SELECT COUNT(*) FROM rooms WHERE building_id=? AND lang='en' AND is_active=1",
        (row["building_id"],),
    ).fetchone()[0]
    lab_count = conn.execute(
        "SELECT COUNT(*) FROM labs WHERE building_id=? AND lang='en' AND is_active=1",
        (row["building_id"],),
    ).fetchone()[0]
    dept_count = conn.execute(
        "SELECT COUNT(*) FROM departments WHERE building=? AND lang='en' AND is_active=1",
        (row["building_id"],),
    ).fetchone()[0]
    parts.append(f"Contains: {room_count} rooms, {lab_count} labs, {dept_count} departments")
    return parts


def _dedupe_keep_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if not line or line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique
