from app.pipeline.arabic_query_understander import understand_arabic


def test_strips_location_question_prefix_and_suffix() -> None:
    result = understand_arabic("عايز أعرف معمل الروبوتيكس فين")

    assert "معمل الروبوتيكس" in result.best_entity
    assert result.query_type == "location"


def test_strips_tell_me_about_department() -> None:
    result = understand_arabic("ممكن تقولي عن قسم الحاسبات")

    assert "قسم الحاسبات" in result.best_entity


def test_navigation_query_type() -> None:
    result = understand_arabic("خدني لمكتب التسجيل")

    assert result.query_type == "navigation"
    assert result.best_entity == "مكتب التسجيل"


def test_person_prefix_detected() -> None:
    result = understand_arabic("الدكتور أحمد مكتبه فين")

    assert result.has_person_prefix is True
    assert "أحمد" in result.best_entity


def test_router_entity_trusted_above_threshold() -> None:
    result = understand_arabic("فين المعمل", router_entity="معمل الروبوتات", router_confidence=0.8)

    assert result.best_entity == "معمل الروبوتات"


def test_router_entity_ignored_below_threshold() -> None:
    result = understand_arabic("فين معمل الروبوتات", router_entity="مكتب التسجيل", router_confidence=0.6)

    assert result.best_entity != "مكتب التسجيل"
    assert "معمل الروبوتات" in result.best_entity
