"""
Navigator - CSV Ingestion Engine
Reads staff-editable CSV files and syncs them into SQLite.

Rules:
- CSV files are import sources only. SQLite is the runtime truth.
- Sync is idempotent — safe to run on every boot.
- Malformed rows are skipped and logged, never silently dropped.
- FTS indexes are rebuilt after every successful sync.
"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.storage.db import get_db
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def normalize_for_search(text: str | None) -> str:
    return normalize(text).lower()


def str_or_none(text: str | None) -> str | None:
    v = normalize(text)
    return v if v else None


# ─────────────────────────────────────────────────────────────────────────────
# Sync Log Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _start_sync_log(conn: sqlite3.Connection, source_name: str) -> int:
    cur = conn.execute(
        "INSERT INTO sync_log (source_name, started_at, status) VALUES (?, ?, 'running');",
        (source_name, datetime.now().isoformat()),
    )
    conn.commit()
    return cur.lastrowid


def _finish_sync_log(
    conn: sqlite3.Connection,
    log_id: int,
    rows_seen: int,
    rows_upserted: int,
    rows_skipped: int,
    rows_errored: int,
    status: str = "ok",
    error_message: str | None = None,
) -> None:
    conn.execute(
        """UPDATE sync_log
           SET finished_at=?, rows_seen=?, rows_upserted=?, rows_skipped=?,
               rows_errored=?, status=?, error_message=?
           WHERE id=?;""",
        (
            datetime.now().isoformat(),
            rows_seen, rows_upserted, rows_skipped,
            rows_errored, status, error_message, log_id,
        ),
    )
    conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Required columns per CSV
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "departments":  ["code", "name"],
    "locations":    ["code", "name"],
    "staff":        ["full_name"],
    "office_hours": ["staff_full_name", "weekday", "start_time", "end_time"],
    "facilities":   ["code", "name"],
    "aliases":      ["canonical_type", "canonical_id", "alias_text"],
    "navigation_targets": ["target_type", "canonical_id", "nav_code"],
}
_NAV_TARGET_TABLES = {
    "location": "locations",
    "department": "departments",
    "facility": "facilities",
}


def _validate_row(row: dict, required: list[str], line: int, source: str) -> bool:
    for col in required:
        if not row.get(col, "").strip():
            logger.warning("csv_missing_field", source=source, line=line, field=col)
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-table sync functions
# ─────────────────────────────────────────────────────────────────────────────

def _sync_departments(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["departments"], i, source):
            skipped += 1
            continue
        try:
            conn.execute(
                """INSERT INTO departments
                       (code, name, building, floor, room, description, head_name, contact_email, is_active, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,1,datetime('now'))
                   ON CONFLICT(code) DO UPDATE SET
                       name=excluded.name, building=excluded.building,
                       floor=excluded.floor, room=excluded.room,
                       description=excluded.description,
                       head_name=excluded.head_name,
                       contact_email=excluded.contact_email,
                       updated_at=excluded.updated_at;""",
                (
                    normalize(row["code"]),
                    normalize(row["name"]),
                    str_or_none(row.get("building")),
                    str_or_none(row.get("floor")),
                    str_or_none(row.get("room")),
                    str_or_none(row.get("description")),
                    str_or_none(row.get("head_name")),
                    str_or_none(row.get("contact_email")),
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_locations(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["locations"], i, source):
            skipped += 1
            continue
        try:
            dept_id = None
            dept_code = str_or_none(row.get("department_code"))
            if dept_code:
                r = conn.execute("SELECT id FROM departments WHERE code=?;", (dept_code,)).fetchone()
                if r:
                    dept_id = r["id"]

            conn.execute(
                """INSERT INTO locations
                       (code, name, building, floor, room, description, department_id, map_node, is_active, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,1,datetime('now'))
                   ON CONFLICT(code) DO UPDATE SET
                       name=excluded.name, building=excluded.building,
                       floor=excluded.floor, room=excluded.room,
                       description=excluded.description,
                       department_id=excluded.department_id,
                       map_node=excluded.map_node,
                       updated_at=excluded.updated_at;""",
                (
                    normalize(row["code"]),
                    normalize(row["name"]),
                    str_or_none(row.get("building")),
                    str_or_none(row.get("floor")),
                    str_or_none(row.get("room")),
                    str_or_none(row.get("description")),
                    dept_id,
                    str_or_none(row.get("map_node")),
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_staff(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["staff"], i, source):
            skipped += 1
            continue
        try:
            dept_id = None
            dept_code = str_or_none(row.get("department_code"))
            if dept_code:
                r = conn.execute("SELECT id FROM departments WHERE code=?;", (dept_code,)).fetchone()
                if r:
                    dept_id = r["id"]

            office_id = None
            office_code = str_or_none(row.get("office_location_code"))
            if office_code:
                r = conn.execute("SELECT id FROM locations WHERE code=?;", (office_code,)).fetchone()
                if r:
                    office_id = r["id"]

            existing = conn.execute(
                "SELECT id FROM staff WHERE full_name=?;",
                (normalize(row["full_name"]),)
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE staff SET title=?, department_id=?, office_location_id=?,
                       contact_notes=?, is_active=1, updated_at=datetime('now')
                       WHERE id=?;""",
                    (
                        str_or_none(row.get("title")),
                        dept_id, office_id,
                        str_or_none(row.get("contact_notes")),
                        existing["id"],
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO staff
                           (full_name, title, department_id, office_location_id, contact_notes, is_active)
                       VALUES (?,?,?,?,?,1);""",
                    (
                        normalize(row["full_name"]),
                        str_or_none(row.get("title")),
                        dept_id, office_id,
                        str_or_none(row.get("contact_notes")),
                    ),
                )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_office_hours(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["office_hours"], i, source):
            skipped += 1
            continue
        try:
            staff = conn.execute(
                "SELECT id FROM staff WHERE full_name=?;",
                (normalize(row["staff_full_name"]),)
            ).fetchone()
            if not staff:
                logger.warning("csv_staff_not_found", source=source, line=i, name=row["staff_full_name"])
                skipped += 1
                continue

            conn.execute(
                "DELETE FROM office_hours WHERE staff_id=? AND weekday=?;",
                (staff["id"], normalize(row["weekday"])),
            )
            conn.execute(
                """INSERT INTO office_hours
                       (staff_id, weekday, start_time, end_time, notes, source_version)
                   VALUES (?,?,?,?,?,?);""",
                (
                    staff["id"],
                    normalize(row["weekday"]),
                    normalize(row["start_time"]),
                    normalize(row["end_time"]),
                    str_or_none(row.get("notes")),
                    str_or_none(row.get("source_version")),
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_facilities(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["facilities"], i, source):
            skipped += 1
            continue
        try:
            conn.execute(
                """INSERT INTO facilities
                       (code, name, category, building, floor, room, description, is_active, updated_at)
                   VALUES (?,?,?,?,?,?,?,1,datetime('now'))
                   ON CONFLICT(code) DO UPDATE SET
                       name=excluded.name, category=excluded.category,
                       building=excluded.building, floor=excluded.floor,
                       room=excluded.room, description=excluded.description,
                       updated_at=excluded.updated_at;""",
                (
                    normalize(row["code"]),
                    normalize(row["name"]),
                    str_or_none(row.get("category")),
                    str_or_none(row.get("building")),
                    str_or_none(row.get("floor")),
                    str_or_none(row.get("room")),
                    str_or_none(row.get("description")),
                ),
            )
            upserted += 1
        except sqlite3.Error as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_aliases(conn: sqlite3.Connection, rows: list[dict], source: str) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["aliases"], i, source):
            skipped += 1
            continue
        try:
            alias_text = normalize(row["alias_text"])
            normalized_alias = normalize_for_search(alias_text)
            canonical_type = normalize(row["canonical_type"])
            canonical_id = int(row["canonical_id"])

            existing = conn.execute(
                "SELECT id FROM aliases WHERE canonical_type=? AND canonical_id=? AND normalized_alias=?;",
                (canonical_type, canonical_id, normalized_alias),
            ).fetchone()

            if not existing:
                conn.execute(
                    "INSERT INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias) VALUES (?,?,?,?);",
                    (canonical_type, canonical_id, alias_text, normalized_alias),
                )
            upserted += 1
        except (sqlite3.Error, ValueError) as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1
    return upserted, skipped, errored


def _sync_navigation_targets(
    conn: sqlite3.Connection,
    rows: list[dict],
    source: str,
) -> tuple[int, int, int]:
    upserted = skipped = errored = 0
    for i, row in enumerate(rows, start=2):
        if not _validate_row(row, REQUIRED_COLUMNS["navigation_targets"], i, source):
            skipped += 1
            continue

        try:
            target_type = normalize_for_search(row["target_type"])
            if target_type not in _NAV_TARGET_TABLES:
                logger.warning("csv_invalid_target_type", source=source, line=i, target_type=target_type)
                skipped += 1
                continue

            canonical_id = int(row["canonical_id"])
            target_exists = conn.execute(
                f"SELECT id FROM {_NAV_TARGET_TABLES[target_type]} WHERE id=?;",
                (canonical_id,),
            ).fetchone()
            if not target_exists:
                logger.warning(
                    "csv_navigation_target_missing_canonical",
                    source=source,
                    line=i,
                    target_type=target_type,
                    canonical_id=canonical_id,
                )
                skipped += 1
                continue

            nav_code = normalize(row["nav_code"])
            safety_notes = str_or_none(row.get("safety_notes"))
            existing = conn.execute(
                "SELECT id FROM navigation_targets WHERE target_type=? AND canonical_id=?;",
                (target_type, canonical_id),
            ).fetchone()

            if existing:
                conn.execute(
                    """UPDATE navigation_targets
                       SET nav_code=?, safety_notes=?, updated_at=datetime('now')
                       WHERE id=?;""",
                    (nav_code, safety_notes, existing["id"]),
                )
            else:
                conn.execute(
                    """INSERT INTO navigation_targets (target_type, canonical_id, nav_code, safety_notes, updated_at)
                       VALUES (?,?,?,?,datetime('now'))
                       ON CONFLICT(nav_code) DO UPDATE SET
                           target_type=excluded.target_type,
                           canonical_id=excluded.canonical_id,
                           safety_notes=excluded.safety_notes,
                           updated_at=excluded.updated_at;""",
                    (target_type, canonical_id, nav_code, safety_notes),
                )
            upserted += 1
        except (sqlite3.Error, ValueError) as exc:
            logger.error("csv_row_error", source=source, line=i, error=str(exc))
            errored += 1

    return upserted, skipped, errored


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_SYNC_HANDLERS = {
    "departments":  _sync_departments,
    "locations":    _sync_locations,
    "staff":        _sync_staff,
    "office_hours": _sync_office_hours,
    "facilities":   _sync_facilities,
    "aliases":      _sync_aliases,
    "navigation_targets": _sync_navigation_targets,
}

# Order matters: departments before locations, locations before staff
_SYNC_ORDER = [
    "departments",
    "locations",
    "staff",
    "office_hours",
    "facilities",
    "aliases",
    "navigation_targets",
]


def sync_all_csvs() -> dict[str, dict]:
    """
    Sync all CSV files in the configured directory into SQLite.
    Returns a summary dict of results per source.
    Idempotent — safe to call on every boot.
    """
    cfg = get_settings()
    csv_dir = Path(cfg.csv_data_dir)
    conn = get_db()
    results: dict[str, dict] = {}

    if not csv_dir.exists():
        logger.warning("csv_dir_missing", path=str(csv_dir))
        return results

    logger.info("csv_sync_start", csv_dir=str(csv_dir))

    for name in _SYNC_ORDER:
        csv_path = csv_dir / f"{name}.csv"
        if not csv_path.exists():
            logger.debug("csv_file_not_found", file=f"{name}.csv")
            continue

        log_id = _start_sync_log(conn, name)
        try:
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            with conn:
                upserted, skipped, errored = _SYNC_HANDLERS[name](conn, rows, name)

            status = "ok" if errored == 0 else "partial"
            _finish_sync_log(conn, log_id, len(rows), upserted, skipped, errored, status)
            results[name] = dict(rows_seen=len(rows), upserted=upserted,
                                 skipped=skipped, errored=errored, status=status)
            logger.info("csv_file_synced", source=name, rows_seen=len(rows),
                        upserted=upserted, skipped=skipped, errored=errored)

        except Exception as exc:
            _finish_sync_log(conn, log_id, 0, 0, 0, 0, "error", str(exc))
            logger.error("csv_sync_failed", source=name, error=str(exc))
            results[name] = {"status": "error", "error": str(exc)}

    from app.storage.schema import rebuild_fts_indexes
    rebuild_fts_indexes()

    logger.info("csv_sync_all_done", sources=list(results.keys()))
    return results
