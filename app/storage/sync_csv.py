"""
Navigator - Bilingual CSV Ingestion Engine

Reads from two source directories:
  data/csv_english/ -> English campus data
  data/csv_arabic/  -> Arabic campus data

Each CSV type has a ColumnMap that translates actual column headers into a
canonical internal key so handler logic stays language-agnostic.
"""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.config import get_settings
from app.retrieval.search import normalize_query
from app.storage.db import get_db
from app.storage.schema import rebuild_fts
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _norm(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def _norm_search(text: str | None, lang: str) -> str:
    value = _norm(text)
    if not value:
        return ""
    return value.lower() if lang == "en" else value


def _opt(text: str | None) -> str | None:
    value = _norm(text)
    return value or None


@dataclass(frozen=True)
class ColumnMap:
    """Maps canonical key -> actual CSV column name for one file variant."""

    mapping: dict[str, str]

    def get(self, row: dict[str, str], key: str, fallback: str = "") -> str:
        column = self.mapping.get(key, key)
        return _norm(row.get(column, fallback))

    def opt(self, row: dict[str, str], key: str) -> str | None:
        value = self.get(row, key)
        return value or None


_DEPT_MAP_EN = ColumnMap(
    {
        "code": "department_id",
        "name": "department_name",
        "building": "building_id",
        "floor": "floor_id",
        "room": "office_room_id",
        "head_room": "head_office_room_id",
    }
)
_DEPT_MAP_AR = ColumnMap(
    {
        "code": "معرف_القسم",
        "name": "اسم_القسم",
        "building": "معرف_المبنى",
        "floor": "معرف_الطابق",
        "room": "معرف_غرفة_المكتب",
        "head_room": "معرف_غرفة_رئيس_القسم",
    }
)

_BUILDINGS_MAP_EN = ColumnMap(
    {
        "building_id": "building_id",
        "building_name": "building_name",
        "description": "description",
    }
)
_BUILDINGS_MAP_AR = ColumnMap(
    {
        "building_id": "معرف_المبنى",
        "building_name": "اسم_المبنى",
        "description": "الوصف",
    }
)

_ROOMS_MAP_EN = ColumnMap(
    {
        "building_id": "building_id",
        "floor_id": "floor_id",
        "room_name": "room_name",
        "room_type": "room_type",
        "room_number": "room_number",
    }
)
_ROOMS_MAP_AR = ColumnMap(
    {
        "building_id": "معرف_المبنى",
        "floor_id": "معرف_الطابق",
        "room_name": "اسم_الغرفة",
        "room_type": "نوع_الغرفة",
        "room_number": "رقم_الغرفة",
    }
)

_LABS_MAP_EN = ColumnMap(
    {
        "lab_group_id": "lab_id",
        "lab_name": "lab_name",
        "building_id": "building_id",
        "floor_id": "floor_id",
        "room_id": "room_id",
        "status": "status",
    }
)
_LABS_MAP_AR = ColumnMap(
    {
        "lab_group_id": "معرف_المعمل",
        "lab_name": "اسم_المعمل",
        "building_id": "معرف_المبنى",
        "floor_id": "معرف_الطابق",
        "room_id": "معرف_الغرفة",
        "status": "الحالة",
    }
)

_FLOORS_MAP_EN = ColumnMap(
    {
        "building_id": "building_id",
        "floor_number": "floor_number",
        "floor_name": "floor_name",
    }
)
_FLOORS_MAP_AR = ColumnMap(
    {
        "building_id": "معرف_المبنى",
        "floor_number": "رقم_الطابق",
        "floor_name": "اسم_الطابق",
    }
)

_LANDMARKS_MAP_EN = ColumnMap(
    {
        "landmark_name": "landmark_name",
        "building_id": "building_id",
        "floor_id": "floor_id",
        "description": "description",
    }
)
_LANDMARKS_MAP_AR = ColumnMap(
    {
        "landmark_name": "اسم_المعلم",
        "building_id": "معرف_المبنى",
        "floor_id": "معرف_الطابق",
        "description": "الوصف",
    }
)

_STAFF_MAP_EN = ColumnMap(
    {
        "staff_id": "staff_id",
        "full_name": "staff_name",
        "title": "staff_role",
        "department_code": "department_id",
        "office_room": "office_room_id",
        "availability": "availability_status",
    }
)
_STAFF_MAP_AR = ColumnMap(
    {
        "staff_id": "معرف الموظف",
        "full_name": "اسم الموظف",
        "title": "وظيفة الموظف",
        "department_code": "معرف القسم",
        "office_room": "معرف غرفة المكتب",
        "availability": "حالة التوفر",
    }
)

_OFFICE_HOURS_MAP_EN = ColumnMap(
    {
        "staff_id": "staff_id",
        "weekday": "day_of_week",
        "start_time": "start_time",
        "end_time": "end_time",
        "source_version": "office_hours_id",
    }
)
_OFFICE_HOURS_MAP_AR = ColumnMap(
    {
        "staff_id": "معرف الموظفين",
        "weekday": "يوم من أيام الأسبوع",
        "start_time": "وقت البدء",
        "end_time": "وقت الانتهاء",
        "source_version": "معرف ساعات العمل",
    }
)

_MEMBERS_MAP_EN = ColumnMap(
    {
        "member_id": "member_id",
        "full_name": "full_name",
        "role": "role",
        "team": "team",
        "bio": "bio",
    }
)
_MEMBERS_MAP_AR = ColumnMap(
    {
        "member_id": "معرف_العضو",
        "full_name": "الاسم_الكامل",
        "role": "الدور",
        "team": "الفريق",
        "bio": "السيرة",
    }
)

_ALIASES_MAP_EN = ColumnMap(
    {
        "canonical_type": "canonical_type",
        "canonical_name": "canonical_name",
        "alias_text": "alias_text",
        "lang": "lang",
    }
)
_ALIASES_MAP_AR = _ALIASES_MAP_EN

_NAV_PATHS_MAP_EN = ColumnMap(
    {
        "target_type": "target_type",
        "canonical_name": "canonical_name",
        "nav_code": "nav_code",
        "safety_notes": "safety_notes",
    }
)
_NAV_PATHS_MAP_AR = _NAV_PATHS_MAP_EN


def _start_log(conn: sqlite3.Connection, source_name: str, lang: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO sync_log (source_name, lang, started_at, status) VALUES (?, ?, ?, 'running')",
        (source_name, lang, datetime.now().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def _finish_log(
    conn: sqlite3.Connection,
    log_id: int,
    seen: int,
    upserted: int,
    skipped: int,
    errored: int,
    status: str = "ok",
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE sync_log
        SET finished_at=?, rows_seen=?, rows_upserted=?, rows_skipped=?,
            rows_errored=?, status=?, error_message=?
        WHERE id=?
        """,
        (datetime.now().isoformat(), seen, upserted, skipped, errored, status, error, log_id),
    )
    conn.commit()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _sync_departments(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        code = cm.get(row, "code")
        name = cm.get(row, "name")
        if not code or not name:
            logger.warning("sync_department_missing_required", lang=lang, line=line_number)
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO departments (code, name, building, floor, room, head_room, lang, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
                ON CONFLICT(code, lang) DO UPDATE SET
                    name=excluded.name,
                    building=excluded.building,
                    floor=excluded.floor,
                    room=excluded.room,
                    head_room=excluded.head_room,
                    updated_at=excluded.updated_at
                """,
                (
                    code,
                    name,
                    cm.opt(row, "building"),
                    cm.opt(row, "floor"),
                    cm.opt(row, "room"),
                    cm.opt(row, "head_room"),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_department_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_buildings(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        building_id = cm.get(row, "building_id")
        building_name = cm.get(row, "building_name")
        if not building_id or not building_name:
            logger.warning("sync_building_missing_required", lang=lang, line=line_number)
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO buildings (building_id, building_name, description, lang, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(building_id, lang) DO UPDATE SET
                    building_name=excluded.building_name,
                    description=excluded.description,
                    updated_at=excluded.updated_at,
                    is_active=1
                """,
                (building_id, building_name, cm.opt(row, "description"), lang),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_building_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_rooms(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        room_number = cm.get(row, "room_number")
        room_name = cm.get(row, "room_name")
        building_id = cm.get(row, "building_id")
        if not room_number or not room_name or not building_id:
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO rooms (room_number, room_name, room_type, building_id, floor_id, lang, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, datetime('now'))
                ON CONFLICT(building_id, room_number, lang) DO UPDATE SET
                    room_name=COALESCE(NULLIF(rooms.room_name, ''), excluded.room_name),
                    room_type=excluded.room_type,
                    floor_id=excluded.floor_id,
                    updated_at=excluded.updated_at
                """,
                (
                    room_number,
                    room_name,
                    cm.opt(row, "room_type"),
                    building_id,
                    cm.opt(row, "floor_id"),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_room_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_labs(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        lab_name = cm.get(row, "lab_name")
        building_id = cm.get(row, "building_id")
        room_id = cm.opt(row, "room_id")
        if not lab_name or not building_id:
            skipped += 1
            continue
        group_raw = cm.opt(row, "lab_group_id")
        group_id = int(group_raw) if group_raw and group_raw.isdigit() else None
        try:
            existing = conn.execute(
                """
                SELECT id FROM labs
                WHERE lab_name=? AND COALESCE(room_id, '')=COALESCE(?, '') AND lang=?
                """,
                (lab_name, room_id, lang),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE labs
                    SET lab_group_id=?, building_id=?, floor_id=?, room_id=?, status=?, updated_at=datetime('now')
                    WHERE id=?
                    """,
                    (
                        group_id,
                        building_id,
                        cm.opt(row, "floor_id"),
                        room_id,
                        cm.opt(row, "status") or "Open",
                        existing["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO labs (lab_group_id, lab_name, building_id, floor_id, room_id, status, lang, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
                    """,
                    (
                        group_id,
                        lab_name,
                        building_id,
                        cm.opt(row, "floor_id"),
                        room_id,
                        cm.opt(row, "status") or "Open",
                        lang,
                    ),
                )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_lab_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_floors(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        building_id = cm.get(row, "building_id")
        floor_number = cm.get(row, "floor_number")
        if not building_id or not floor_number:
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO floors (building_id, floor_number, floor_name, lang, is_active, updated_at)
                VALUES (?, ?, ?, ?, 1, datetime('now'))
                ON CONFLICT(building_id, floor_number, lang) DO UPDATE SET
                    floor_name=excluded.floor_name,
                    updated_at=excluded.updated_at
                """,
                (building_id, floor_number, cm.opt(row, "floor_name"), lang),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_floor_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_landmarks(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        landmark_name = cm.get(row, "landmark_name")
        if not landmark_name:
            skipped += 1
            continue
        try:
            existing = conn.execute(
                """
                SELECT id FROM landmarks
                WHERE landmark_name=? AND COALESCE(building_id, '')=COALESCE(?, '') AND lang=?
                """,
                (landmark_name, cm.opt(row, "building_id"), lang),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE landmarks
                    SET building_id=?, floor_id=?, description=?, updated_at=datetime('now')
                    WHERE id=?
                    """,
                    (
                        cm.opt(row, "building_id"),
                        cm.opt(row, "floor_id"),
                        cm.opt(row, "description"),
                        existing["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO landmarks (landmark_name, building_id, floor_id, description, lang, is_active, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, datetime('now'))
                    """,
                    (
                        landmark_name,
                        cm.opt(row, "building_id"),
                        cm.opt(row, "floor_id"),
                        cm.opt(row, "description"),
                        lang,
                    ),
                )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_landmark_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_staff_en(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        source_staff_id = cm.get(row, "staff_id")
        full_name = cm.get(row, "full_name")
        if not source_staff_id or not full_name:
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO staff (source_staff_id, full_name, title, department_code, office_room, contact_notes, lang, is_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
                ON CONFLICT(full_name) DO UPDATE SET
                    source_staff_id=excluded.source_staff_id,
                    title=excluded.title,
                    department_code=excluded.department_code,
                    office_room=excluded.office_room,
                    contact_notes=excluded.contact_notes,
                    updated_at=excluded.updated_at
                """,
                (
                    source_staff_id,
                    full_name,
                    cm.opt(row, "title"),
                    cm.opt(row, "department_code"),
                    cm.opt(row, "office_room"),
                    cm.opt(row, "availability"),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_staff_en_error", line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_staff_ar_aliases(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        source_staff_id = cm.get(row, "staff_id")
        alias_name = cm.get(row, "full_name")
        if not source_staff_id or not alias_name:
            skipped += 1
            continue
        staff_row = conn.execute(
            "SELECT id FROM staff WHERE source_staff_id=?",
            (source_staff_id,),
        ).fetchone()
        if not staff_row:
            logger.warning("sync_staff_ar_missing_canonical_staff", staff_id=source_staff_id, line=line_number)
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, lang)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(canonical_type, canonical_id, normalized_alias, lang) DO UPDATE SET
                    alias_text=excluded.alias_text
                """,
                (
                    "staff",
                    staff_row["id"],
                    alias_name,
                    normalize_query(alias_name, lang),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_staff_ar_alias_error", line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_office_hours(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        source_staff_id = cm.get(row, "staff_id")
        weekday = cm.get(row, "weekday")
        start_time = cm.get(row, "start_time")
        end_time = cm.get(row, "end_time")
        if not source_staff_id or not weekday or not start_time or not end_time:
            skipped += 1
            continue
        staff_row = conn.execute(
            "SELECT full_name FROM staff WHERE source_staff_id=?",
            (source_staff_id,),
        ).fetchone()
        if not staff_row:
            logger.warning("sync_office_hours_missing_staff", lang=lang, staff_id=source_staff_id, line=line_number)
            skipped += 1
            continue
        try:
            conn.execute(
                """
                DELETE FROM office_hours
                WHERE staff_full_name=? AND weekday=? AND start_time=? AND end_time=?
                """,
                (staff_row["full_name"], weekday, start_time, end_time),
            )
            conn.execute(
                """
                INSERT INTO office_hours (staff_id, staff_full_name, weekday, start_time, end_time, notes, source_version, lang, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    source_staff_id,
                    staff_row["full_name"],
                    weekday,
                    start_time,
                    end_time,
                    None,
                    cm.opt(row, "source_version"),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_office_hours_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_aliases(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    """Sync CSV-backed aliases by resolving canonical names to IDs."""
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        canonical_type = cm.get(row, "canonical_type")
        canonical_name = cm.get(row, "canonical_name")
        alias_text = cm.get(row, "alias_text")
        alias_lang = cm.get(row, "lang", lang) or lang
        if not canonical_type or not canonical_name or not alias_text:
            skipped += 1
            continue
        resolved = _resolve_canonical_for_alias(conn, canonical_type, canonical_name, alias_lang)
        if resolved is None:
            if alias_lang != "en":
                logger.debug(
                    "sync_alias_missing_canonical_skipped_after_en_fallback",
                    canonical_type=canonical_type,
                    canonical_name=canonical_name,
                    line=line_number,
                )
                skipped += 1
                continue
            logger.warning(
                "sync_alias_missing_canonical",
                canonical_type=canonical_type,
                canonical_name=canonical_name,
                line=line_number,
            )
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, alias_text_norm, lang)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_type, canonical_id, normalized_alias, lang) DO UPDATE SET
                    alias_text=excluded.alias_text,
                    alias_text_norm=excluded.alias_text_norm
                """,
                (
                    resolved[0],
                    resolved[1],
                    alias_text,
                    normalize_query(alias_text, alias_lang),
                    normalize_query(alias_text, alias_lang),
                    alias_lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_alias_error", line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_members(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        full_name = cm.get(row, "full_name")
        if not full_name:
            logger.warning("sync_member_missing_required", lang=lang, line=line_number)
            skipped += 1
            continue
        try:
            conn.execute(
                """
                INSERT INTO members (member_id, full_name, role, team, bio, lang, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(full_name, lang) DO UPDATE SET
                    member_id=excluded.member_id,
                    role=excluded.role,
                    team=excluded.team,
                    bio=excluded.bio,
                    updated_at=excluded.updated_at,
                    is_active=1
                """,
                (
                    cm.opt(row, "member_id"),
                    full_name,
                    cm.opt(row, "role"),
                    cm.opt(row, "team") or "Innovtronics",
                    cm.opt(row, "bio"),
                    lang,
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_member_error", lang=lang, line=line_number, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_navigation_paths(
    conn: sqlite3.Connection,
    rows: list[dict[str, str]],
    lang: str,
    cm: ColumnMap,
) -> tuple[int, int, int]:
    """
    Map canonical entity names to nav codes in navigation_targets.
    Resolves canonical_id by looking up the name in the appropriate table.
    """
    upserted = skipped = errored = 0
    for line_number, row in enumerate(rows, start=2):
        target_type = cm.get(row, "target_type").lower().strip()
        canonical_name = cm.get(row, "canonical_name")
        nav_code = cm.get(row, "nav_code")

        if not target_type or not canonical_name or not nav_code:
            skipped += 1
            continue

        canonical_id = _resolve_canonical_id(conn, target_type, canonical_name)
        if canonical_id is None:
            logger.warning(
                "nav_path_canonical_not_found",
                target_type=target_type,
                canonical_name=canonical_name,
                line=line_number,
            )
            skipped += 1
            continue

        try:
            conn.execute(
                """
                INSERT INTO navigation_targets
                    (target_type, canonical_id, nav_code, safety_notes, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(target_type, canonical_id) DO UPDATE SET
                    nav_code=excluded.nav_code,
                    safety_notes=excluded.safety_notes,
                    updated_at=excluded.updated_at
                """,
                (
                    target_type,
                    canonical_id,
                    nav_code,
                    cm.opt(row, "safety_notes"),
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("sync_nav_path_error", lang=lang, line=line_number, error=str(exc))
            errored += 1

    return upserted, skipped, errored


def _resolve_canonical_id(
    conn: sqlite3.Connection,
    target_type: str,
    canonical_name: str,
) -> int | None:
    """
    Resolve a canonical name to its integer primary key.
    Tries exact match first, alternate IDs next, then a broad name match.
    """
    table_lookup = {
        "room": ("rooms", "room_name", "room_number"),
        "lab": ("labs", "lab_name", None),
        "department": ("departments", "name", "code"),
        "landmark": ("landmarks", "landmark_name", None),
        "building": ("buildings", "building_name", "building_id"),
        "staff": ("staff", "full_name", None),
        "member": ("members", "full_name", "team"),
    }
    if target_type not in table_lookup:
        return None

    table, name_col, alt_col = table_lookup[target_type]

    row = conn.execute(
        f"""
        SELECT id FROM {table}
        WHERE LOWER({name_col})=LOWER(?)
        ORDER BY CASE COALESCE(lang, 'en') WHEN 'en' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (canonical_name,),
    ).fetchone()
    if row:
        return row["id"]

    if alt_col:
        row = conn.execute(
            f"""
            SELECT id FROM {table}
            WHERE LOWER({alt_col})=LOWER(?)
            ORDER BY CASE COALESCE(lang, 'en') WHEN 'en' THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (canonical_name,),
        ).fetchone()
        if row:
            return row["id"]

    row = conn.execute(
        f"""
        SELECT id FROM {table}
        WHERE LOWER({name_col}) LIKE LOWER(?)
        ORDER BY CASE COALESCE(lang, 'en') WHEN 'en' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (f"%{canonical_name}%",),
    ).fetchone()
    return row["id"] if row else None


def _resolve_canonical_for_alias(
    conn: sqlite3.Connection,
    canonical_type: str,
    canonical_name: str,
    lang: str = "en",
) -> tuple[str, int] | None:
    mapping = {
        "room": ("rooms", "room_name"),
        "lab": ("labs", "lab_name"),
        "department": ("departments", "name"),
        "landmark": ("landmarks", "landmark_name"),
        "staff": ("staff", "full_name"),
        "building": ("buildings", "building_name"),
        "member": ("members", "full_name"),
    }
    search_order = [canonical_type] + [
        item
        for item in ("room", "lab", "department", "landmark", "staff", "building", "member")
        if item != canonical_type
    ]
    for entity_type in search_order:
        if entity_type not in mapping:
            continue
        resolved = _find_canonical_id(conn, entity_type, canonical_name, lang)
        if resolved:
            return resolved
    return None


def _find_canonical_id(
    conn: sqlite3.Connection,
    canonical_type: str,
    canonical_name: str,
    lang: str = "en",
) -> tuple[str, int] | None:
    result = _lookup_canonical(conn, canonical_type, canonical_name, lang)
    if result:
        return result
    if lang != "en":
        result = _lookup_canonical(conn, canonical_type, canonical_name, "en")
        if result:
            logger.debug(
                "alias_canonical_found_via_en_fallback",
                canonical_name=canonical_name,
                original_lang=lang,
            )
            return result
    return None


def _lookup_canonical(
    conn: sqlite3.Connection,
    canonical_type: str,
    canonical_name: str,
    lang: str,
) -> tuple[str, int] | None:
    mapping = {
        "room": ("rooms", "room_name"),
        "lab": ("labs", "lab_name"),
        "department": ("departments", "name"),
        "landmark": ("landmarks", "landmark_name"),
        "staff": ("staff", "full_name"),
        "building": ("buildings", "building_name"),
        "member": ("members", "full_name"),
    }
    if canonical_type not in mapping:
        return None

    table, column = mapping[canonical_type]
    row = conn.execute(
        f"SELECT id FROM {table} WHERE lang=? AND LOWER({column})=LOWER(?) LIMIT 1",
        (lang, canonical_name),
    ).fetchone()
    if row:
        return canonical_type, row["id"]

    first_word = canonical_name.split()[0] if canonical_name.split() else ""
    if len(first_word) < 4:
        return None

    row = conn.execute(
        f"SELECT id FROM {table} WHERE lang=? AND LOWER({column}) LIKE LOWER(?) LIMIT 1",
        (lang, f"%{first_word}%"),
    ).fetchone()
    if row:
        return canonical_type, row["id"]
    return None


@dataclass(frozen=True)
class _CSVSpec:
    handler: Callable[[sqlite3.Connection, list[dict[str, str]], str, ColumnMap], tuple[int, int, int]]
    map_en: ColumnMap
    map_ar: ColumnMap


_SPECS: dict[str, _CSVSpec] = {
    "buildings": _CSVSpec(_sync_buildings, _BUILDINGS_MAP_EN, _BUILDINGS_MAP_AR),
    "floors": _CSVSpec(_sync_floors, _FLOORS_MAP_EN, _FLOORS_MAP_AR),
    "departments": _CSVSpec(_sync_departments, _DEPT_MAP_EN, _DEPT_MAP_AR),
    "rooms": _CSVSpec(_sync_rooms, _ROOMS_MAP_EN, _ROOMS_MAP_AR),
    "labs": _CSVSpec(_sync_labs, _LABS_MAP_EN, _LABS_MAP_AR),
    "landmarks": _CSVSpec(_sync_landmarks, _LANDMARKS_MAP_EN, _LANDMARKS_MAP_AR),
    "staff": _CSVSpec(_sync_staff_en, _STAFF_MAP_EN, _STAFF_MAP_AR),
    "office_hours": _CSVSpec(_sync_office_hours, _OFFICE_HOURS_MAP_EN, _OFFICE_HOURS_MAP_AR),
    "members": _CSVSpec(_sync_members, _MEMBERS_MAP_EN, _MEMBERS_MAP_AR),
    "aliases": _CSVSpec(_sync_aliases, _ALIASES_MAP_EN, _ALIASES_MAP_AR),
    "navigation_paths": _CSVSpec(_sync_navigation_paths, _NAV_PATHS_MAP_EN, _NAV_PATHS_MAP_AR),
}

_SYNC_ORDER = [
    "buildings",
    "floors",
    "departments",
    "rooms",
    "labs",
    "landmarks",
    "staff",
    "office_hours",
    "members",
    "aliases",
    "navigation_paths",
]
_IGNORED_STEMS: frozenset[str] = frozenset()


def _stem_to_entity(stem: str, default_lang: str) -> tuple[str, str] | None:
    lower = stem.lower().rstrip(".")
    for suffix, lang in (("_en", "en"), ("_ar", "ar")):
        if lower.endswith(suffix):
            entity = lower[: -len(suffix)]
            if entity in _SPECS:
                return entity, lang
            if entity in _IGNORED_STEMS:
                return None
    if lower in _SPECS:
        return lower, default_lang
    if lower in _IGNORED_STEMS:
        return None
    return None


def sync_directory(directory: Path, default_lang: str) -> dict[str, dict]:
    """
    Sync one CSV directory. `default_lang` is used when the file stem has no
    _en / _ar suffix, which covers the legacy Arabic office-hours file.
    """
    conn = get_db()
    results: dict[str, dict] = {}
    files_by_entity: dict[str, list[tuple[Path, str]]] = {key: [] for key in _SYNC_ORDER}

    for path in directory.glob("*.csv"):
        parsed = _stem_to_entity(path.stem, default_lang)
        if parsed is None:
            logger.warning("sync_csv_ignored", path=str(path))
            continue
        entity, lang = parsed
        files_by_entity[entity].append((path, lang))

    for entity in _SYNC_ORDER:
        spec = _SPECS[entity]
        for path, lang in files_by_entity[entity]:
            log_id = _start_log(conn, path.stem, lang)
            try:
                rows = _read_csv(path)
                if entity == "staff" and lang == "ar":
                    upserted, skipped, errored = _sync_staff_ar_aliases(conn, rows, lang, spec.map_ar)
                else:
                    col_map = spec.map_ar if lang == "ar" else spec.map_en
                    upserted, skipped, errored = spec.handler(conn, rows, lang, col_map)
                conn.commit()
                status = "ok" if errored == 0 else "partial"
                _finish_log(conn, log_id, len(rows), upserted, skipped, errored, status)
                results[path.stem] = {
                    "lang": lang,
                    "seen": len(rows),
                    "upserted": upserted,
                    "skipped": skipped,
                    "errored": errored,
                }
                logger.info(
                    "sync_csv_done",
                    source=path.stem,
                    lang=lang,
                    upserted=upserted,
                    skipped=skipped,
                    errored=errored,
                )
            except Exception as exc:
                conn.rollback()
                _finish_log(conn, log_id, 0, 0, 0, 0, "error", str(exc))
                logger.error("sync_csv_failed", source=path.stem, error=str(exc))
                results[path.stem] = {"lang": lang, "error": str(exc)}
    return results


def sync_all_csvs() -> dict[str, dict]:
    """Sync both language directories and rebuild FTS indexes."""
    cfg = get_settings()
    combined: dict[str, dict] = {}

    english_dir = Path(cfg.csv_english_dir)
    arabic_dir = Path(cfg.csv_arabic_dir)

    if english_dir.exists():
        combined.update(sync_directory(english_dir, "en"))
    else:
        logger.warning("sync_en_dir_missing", path=str(english_dir))

    if arabic_dir.exists():
        combined.update(sync_directory(arabic_dir, "ar"))
    else:
        logger.warning("sync_ar_dir_missing", path=str(arabic_dir))

    rebuild_fts()
    logger.info("sync_all_complete", sources=list(combined.keys()))
    return combined
