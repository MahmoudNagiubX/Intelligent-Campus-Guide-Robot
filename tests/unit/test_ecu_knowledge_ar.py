from app.retrieval.ecu_knowledge_ar import search_ecu_knowledge_ar


def test_seed_data_loads_and_matches_engineering() -> None:
    result = search_ecu_knowledge_ar("هندسة")

    assert result.found is True
    assert "الهندسة" in result.title or "الهندسة" in result.content


def test_matches_pharmacy() -> None:
    result = search_ecu_knowledge_ar("صيدلة")

    assert result.found is True
    assert "الصيدلة" in result.title or "الصيدلة" in result.content


def test_matches_library() -> None:
    result = search_ecu_knowledge_ar("مكتبة")

    assert result.found is True
    assert "مكتبة" in result.title or "المكتبة" in result.content


def test_unknown_arabic_query_not_found() -> None:
    result = search_ecu_knowledge_ar("نانو روبوتات كوانتم")

    assert result.found is False


def test_fuzzy_faculty_match() -> None:
    result = search_ecu_knowledge_ar("كلية الهندسة")

    assert result.found is True
    assert "الهندسة" in result.title
