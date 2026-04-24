"""
ECU institutional knowledge: fees, GPA, deans, programs, and admissions.

Loaded from data/ecu_institutional.json and cached for the process lifetime.
This data answers Academic_Query intents.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)

_INSTITUTIONAL_PATH = Path("data/ecu_institutional.json")


@lru_cache(maxsize=1)
def _load_institutional_data() -> dict:
    if not _INSTITUTIONAL_PATH.exists():
        logger.warning("ecu_institutional.file_missing")
        return {}
    try:
        data = json.loads(_INSTITUTIONAL_PATH.read_text(encoding="utf-8"))
        logger.info("ecu_institutional.loaded")
        return data
    except Exception as exc:
        logger.error("ecu_institutional.load_failed", error=str(exc))
        return {}


def build_institutional_context(query: str) -> str:
    """Build focused ECU institutional context relevant to an academic query."""
    data = _load_institutional_data()
    if not data:
        return ""

    q = (query or "").lower()
    sections: list[str] = []

    core = data.get("university_core", {})
    sections.append(f"University: {core.get('official_name', 'ECU')}")
    sections.append(f"Founded: {core.get('foundation_year', 'N/A')}")
    sections.append(f"Legal basis: {core.get('legal_foundation', 'N/A')}")
    sections.append(f"Model: {core.get('university_model', 'N/A')}")
    sections.append(f"Website: {core.get('official_website', 'ecu.edu.eg')}")
    campuses = core.get("campuses", [])
    if campuses:
        sections.append(f"Campuses: {', '.join(campuses)}")

    if any(w in q for w in ["president", "dean", "head", "leader", "who runs", "management", "board"]):
        leadership = core.get("macro_leadership", {})
        sections.append(f"\nUniversity President: {leadership.get('university_president', 'N/A')}")
        sections.append(f"Vice President: {leadership.get('university_vice_president', 'N/A')}")
        sections.append(f"Board of Trustees Head: {leadership.get('board_of_trustees_head', 'N/A')}")

        eng = data.get("faculty_of_engineering_and_technology", {}).get("executive_leadership_big_heads", {})
        dean = eng.get("dean_of_engineering_and_technology", {})
        if dean:
            sections.append(f"\nDean of Engineering: {dean.get('name', 'N/A')}")
            sections.append(f"Research: {dean.get('research_focus', 'N/A')}")
            sections.append(f"Recognition: {dean.get('global_recognition', 'N/A')}")

        for faculty, info in core.get("decanal_leadership_non_engineering", {}).items():
            dean_name = info.get("dean") or info.get("acting_dean")
            if dean_name:
                sections.append(f"Dean of {faculty.replace('_', ' ').title()}: {dean_name}")

    if any(w in q for w in ["fee", "cost", "tuition", "payment", "price", "how much", "installment", "money"]):
        fees = core.get("financial_architecture_2023_2024_egp", {})
        sections.append(f"\nPayment: {fees.get('payment_structure', 'N/A')}")
        for faculty, info in fees.get("faculties", {}).items():
            sections.append(f"{faculty.replace('_', ' ').title()}: {info.get('total_annual', 'N/A')} EGP/year")

    if any(w in q for w in ["gpa", "credit", "hour", "graduate", "pass", "fail", "academic system"]):
        framework = data.get("faculty_of_engineering_and_technology", {}).get("pedagogical_framework", {})
        sections.append(f"\nAcademic system: {framework.get('academic_system', 'N/A')}")
        sections.append(f"Credit hours required: {framework.get('total_credit_hours_required', 'N/A')}")
        sections.append(f"Minimum GPA to graduate: {framework.get('minimum_graduation_gpa', 'N/A')}")
        sections.append(f"Credit hour definition: {framework.get('credit_hour_definition', 'N/A')}")

    if any(w in q for w in ["faculty", "faculties", "department", "program", "major", "specializ", "course", "mechatron", "software", "civil", "construction"]):
        fees = core.get("financial_architecture_2023_2024_egp", {})
        if fees.get("faculties"):
            faculty_names = [name.replace("_", " ").title() for name in fees["faculties"]]
            sections.append(f"\nFaculties: {', '.join(faculty_names)}")

        departments = data.get("faculty_of_engineering_and_technology", {}).get("academic_departments", [])
        for dept in departments:
            sections.append(f"\nDepartment: {dept.get('department_name', 'N/A')}")
            if dept.get("acronym"):
                sections.append(f"Code: {dept['acronym']}")
            if dept.get("pedagogical_focus"):
                sections.append(f"Focus: {dept['pedagogical_focus']}")
            lead = dept.get("leadership", {})
            hod = lead.get("head_of_department") or lead.get("charge_daffaires")
            if hod:
                sections.append(f"Head: {hod}")

    if any(w in q for w in ["admission", "apply", "enroll", "register", "requirement", "how to join"]):
        sections.append(f"\nAdmissions: Check {core.get('official_website', 'ecu.edu.eg')} for current requirements.")
        intl = core.get("admissions_framework", {}).get("international_requirements", {})
        if intl:
            sections.append(f"International: {intl.get('general_mandate', 'N/A')}")

    if any(w in q for w in ["lab", "laboratory", "labs"]):
        labs = data.get("faculty_of_engineering_and_technology", {}).get("laboratory_infrastructure", {})
        for category, lab_list in labs.items():
            if lab_list:
                sections.append(f"\n{category.replace('_', ' ').title()}: {', '.join(lab_list)}")

    return "\n".join(sections)
