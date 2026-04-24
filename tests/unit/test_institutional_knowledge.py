"""Test ECU institutional knowledge loading and context selection."""


def test_institutional_data_loads():
    from app.retrieval.ecu_institutional import _load_institutional_data

    _load_institutional_data.cache_clear()
    data = _load_institutional_data()
    assert "university_core" in data
    assert "faculty_of_engineering_and_technology" in data
    _load_institutional_data.cache_clear()


def test_build_context_for_fees():
    from app.retrieval.ecu_institutional import build_institutional_context, _load_institutional_data

    _load_institutional_data.cache_clear()
    ctx = build_institutional_context("how much are engineering fees")
    assert "56000" in ctx or "56,000" in ctx or "EGP" in ctx
    _load_institutional_data.cache_clear()


def test_build_context_for_gpa():
    from app.retrieval.ecu_institutional import build_institutional_context, _load_institutional_data

    _load_institutional_data.cache_clear()
    ctx = build_institutional_context("what gpa do I need to graduate")
    assert "2.0" in ctx
    _load_institutional_data.cache_clear()

