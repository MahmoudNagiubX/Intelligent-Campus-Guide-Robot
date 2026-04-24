from __future__ import annotations

import csv
from pathlib import Path

import pytest

from app.config.settings import get_settings
from app.storage.db import close_db, get_db
from app.storage.schema import bootstrap_schema
from app.storage.sync_csv import _norm_search, _read_csv, sync_all_csvs


def _write_csv(path: Path, rows: list[dict[str, str]], *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def bilingual_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    english_dir = tmp_path / "csv_english"
    arabic_dir = tmp_path / "csv_arabic"
    db_path = tmp_path / "navigator.sqlite"

    close_db()
    get_settings.cache_clear()
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_path))
    monkeypatch.setenv("CSV_DATA_DIR", str(english_dir))
    monkeypatch.setenv("CSV_ENGLISH_DIR", str(english_dir))
    monkeypatch.setenv("CSV_ARABIC_DIR", str(arabic_dir))
    get_settings.cache_clear()

    yield english_dir, arabic_dir

    close_db()
    get_settings.cache_clear()


def test_read_csv_is_bom_safe_for_arabic(tmp_path: Path) -> None:
    csv_path = tmp_path / "rooms_ar.csv"
    _write_csv(
        csv_path,
        [
            {
                "معرف_المبنى": "C",
                "معرف_الطابق": "1",
                "اسم_الغرفة": "معمل الروبوتات",
                "نوع_الغرفة": "Lab",
                "رقم_الغرفة": "C105",
            }
        ],
        encoding="utf-8-sig",
    )

    rows = _read_csv(csv_path)

    assert rows[0]["اسم_الغرفة"] == "معمل الروبوتات"
    assert rows[0]["رقم_الغرفة"] == "C105"


def test_arabic_search_normalization_does_not_lowercase() -> None:
    assert _norm_search("معمل الروبوتات", "ar") == "معمل الروبوتات"
    assert _norm_search("Robotics Lab", "en") == "robotics lab"


def test_sync_all_csvs_loads_bilingual_rows_and_preserves_canonical_staff_rule(
    bilingual_env: tuple[Path, Path],
) -> None:
    english_dir, arabic_dir = bilingual_env

    _write_csv(
        english_dir / "departments_en.csv",
        [
            {
                "department_id": "SET",
                "department_name": "Software Engineering Department",
                "building_id": "C",
                "floor_id": "1",
                "office_room_id": "C111",
                "head_office_room_id": "C112",
            }
        ],
    )
    _write_csv(
        arabic_dir / "departments_ar.csv",
        [
            {
                "معرف_القسم": "SET",
                "اسم_القسم": "قسم هندسة البرمجيات",
                "معرف_المبنى": "C",
                "معرف_الطابق": "1",
                "معرف_غرفة_المكتب": "C111",
                "معرف_غرفة_رئيس_القسم": "C112",
            }
        ],
    )
    _write_csv(
        english_dir / "rooms_en.csv",
        [
            {
                "building_id": "C",
                "floor_id": "1",
                "room_name": "Robotics Lab",
                "room_type": "Lab",
                "room_number": "C105",
            }
        ],
    )
    _write_csv(
        arabic_dir / "rooms_ar.csv",
        [
            {
                "معرف_المبنى": "C",
                "معرف_الطابق": "1",
                "اسم_الغرفة": "معمل الروبوتات",
                "نوع_الغرفة": "Lab",
                "رقم_الغرفة": "C105",
            }
        ],
        encoding="utf-8-sig",
    )
    _write_csv(
        english_dir / "staff_en.csv",
        [
            {
                "staff_id": "1",
                "staff_name": "Dr. Islam Mohamed",
                "staff_role": "Professor",
                "department_id": "SET",
                "office_room_id": "C112",
                "availability_status": "Available",
            }
        ],
    )
    _write_csv(
        arabic_dir / "staff_ar.csv",
        [
            {
                "معرف الموظف": "1",
                "اسم الموظف": "د. إسلام محمد",
                "وظيفة الموظف": "أستاذ",
                "معرف القسم": "SET",
                "معرف غرفة المكتب": "C112",
                "حالة التوفر": "متاح",
            }
        ],
        encoding="utf-8-sig",
    )
    _write_csv(
        english_dir / "office_hours_en.csv",
        [
            {
                "office_hours_id": "OH-1",
                "staff_id": "1",
                "day_of_week": "Sunday",
                "start_time": "10:00",
                "end_time": "12:00",
            }
        ],
    )
    _write_csv(
        arabic_dir / "office_hours.csv",
        [
            {
                "معرف ساعات العمل": "AR-1",
                "معرف الموظفين": "1",
                "يوم من أيام الأسبوع": "الأحد",
                "وقت البدء": "12:00",
                "وقت الانتهاء": "14:00",
            }
        ],
        encoding="utf-8-sig",
    )

    bootstrap_schema()
    results = sync_all_csvs()
    conn = get_db()

    assert results["departments_en"]["upserted"] == 1
    assert results["departments_ar"]["upserted"] == 1
    assert results["rooms_en"]["upserted"] == 1
    assert results["rooms_ar"]["upserted"] == 1
    assert results["staff_en"]["upserted"] == 1
    assert results["staff_ar"]["upserted"] == 1

    department_rows = conn.execute("SELECT code, lang, name FROM departments ORDER BY lang").fetchall()
    assert [(row["code"], row["lang"]) for row in department_rows] == [("SET", "ar"), ("SET", "en")]

    room_rows = conn.execute(
        "SELECT room_number, lang, room_name FROM rooms WHERE room_number='C105' ORDER BY lang"
    ).fetchall()
    assert [(row["room_number"], row["lang"]) for row in room_rows] == [("C105", "ar"), ("C105", "en")]

    staff_rows = conn.execute("SELECT source_staff_id, full_name, lang FROM staff").fetchall()
    assert len(staff_rows) == 1
    assert staff_rows[0]["source_staff_id"] == "1"
    assert staff_rows[0]["full_name"] == "Dr. Islam Mohamed"
    assert staff_rows[0]["lang"] == "en"

    alias_row = conn.execute(
        "SELECT canonical_type, alias_text, lang FROM aliases WHERE canonical_type='staff'"
    ).fetchone()
    assert alias_row["alias_text"] == "د. إسلام محمد"
    assert alias_row["lang"] == "ar"

    office_hours_rows = conn.execute(
        "SELECT staff_full_name, weekday, start_time, end_time FROM office_hours ORDER BY start_time"
    ).fetchall()
    assert {row["staff_full_name"] for row in office_hours_rows} == {"Dr. Islam Mohamed"}
    assert len(office_hours_rows) == 2


def test_sync_handles_new_csv_types_without_ignoring_them(
    bilingual_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    english_dir, arabic_dir = bilingual_env
    warnings: list[str] = []

    _write_csv(
        english_dir / "rooms_en.csv",
        [
            {
                "building_id": "C",
                "floor_id": "1",
                "room_name": "Robotics Lab",
                "room_type": "Lab",
                "room_number": "C105",
            }
        ],
    )
    _write_csv(english_dir / "buildings_en.csv", [{"building_id": "C", "building_name": "Block C"}])
    _write_csv(arabic_dir / "navigation_paths_ar.csv", [{"معرف": "1"}], encoding="utf-8-sig")
    _write_csv(arabic_dir / "members_ar.csv", [{"معرف": "1"}], encoding="utf-8-sig")

    monkeypatch.setattr(
        "app.storage.sync_csv.logger.warning",
        lambda event, **kwargs: warnings.append(f"{event}:{kwargs.get('path', '')}"),
    )

    bootstrap_schema()
    results = sync_all_csvs()

    assert "rooms_en" in results
    assert "buildings_en" in results
    assert "navigation_paths_ar" in results
    assert "members_ar" in results
    assert all("error" not in payload for payload in results.values())
    assert not any(message.startswith("sync_csv_ignored") for message in warnings)


def test_sync_all_csvs_is_idempotent_for_canonical_staff_and_aliases(
    bilingual_env: tuple[Path, Path],
) -> None:
    english_dir, arabic_dir = bilingual_env

    _write_csv(
        english_dir / "staff_en.csv",
        [
            {
                "staff_id": "7",
                "staff_name": "Dr. Sara Ali",
                "staff_role": "Professor",
                "department_id": "SET",
                "office_room_id": "C112",
                "availability_status": "Available",
            }
        ],
    )
    _write_csv(
        arabic_dir / "staff_ar.csv",
        [
            {
                "معرف الموظف": "7",
                "اسم الموظف": "د. سارة علي",
                "وظيفة الموظف": "أستاذ",
                "معرف القسم": "SET",
                "معرف غرفة المكتب": "C112",
                "حالة التوفر": "متاح",
            }
        ],
        encoding="utf-8-sig",
    )

    bootstrap_schema()
    sync_all_csvs()
    sync_all_csvs()
    conn = get_db()

    staff_count = conn.execute("SELECT COUNT(*) FROM staff").fetchone()[0]
    alias_count = conn.execute("SELECT COUNT(*) FROM aliases WHERE canonical_type='staff'").fetchone()[0]

    assert staff_count == 1
    assert alias_count == 1
