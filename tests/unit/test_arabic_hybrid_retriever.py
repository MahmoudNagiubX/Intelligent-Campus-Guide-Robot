from app.pipeline.arabic_query_understander import understand_arabic
from app.retrieval.arabic_hybrid_retriever import retrieve_arabic_hybrid
from app.retrieval.ecu_knowledge_ar import ECUKnowledgeArResult
from app.utils.contracts import RetrievalResult, RetrievalStatus


def _not_found() -> RetrievalResult:
    return RetrievalResult(status=RetrievalStatus.NOT_FOUND)


def test_db_hit_returns_db(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.retrieval.arabic_hybrid_retriever.search",
        lambda *_, **__: RetrievalResult(status=RetrievalStatus.OK, canonical_name="معمل الروبوتات", confidence=0.95),
    )

    result = retrieve_arabic_hybrid(understand_arabic("فين معمل الروبوتات"))

    assert result.answered_by == "db"
    assert result.db_result is not None


def test_db_ambiguous_returns_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.retrieval.arabic_hybrid_retriever.search",
        lambda *_, **__: RetrievalResult(status=RetrievalStatus.AMBIGUOUS, candidates=["أ", "ب"]),
    )

    result = retrieve_arabic_hybrid(understand_arabic("فين المعمل"))

    assert result.answered_by == "clarification"
    assert result.answered_by is not None


def test_db_miss_ecu_hit(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.arabic_hybrid_retriever.search", lambda *_, **__: _not_found())
    monkeypatch.setattr(
        "app.retrieval.arabic_hybrid_retriever.search_ecu_knowledge_ar",
        lambda *_: ECUKnowledgeArResult(found=True, title="كلية الهندسة", content="الهندسة"),
    )

    result = retrieve_arabic_hybrid(understand_arabic("كلية الهندسة"))

    assert result.answered_by == "ecu_web"
    assert result.ecu_result is not None


def test_db_miss_ecu_miss_general(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.arabic_hybrid_retriever.search", lambda *_, **__: _not_found())
    monkeypatch.setattr(
        "app.retrieval.arabic_hybrid_retriever.search_ecu_knowledge_ar",
        lambda *_: ECUKnowledgeArResult(found=False),
    )

    result = retrieve_arabic_hybrid(understand_arabic("شيء غير معروف"))

    assert result.answered_by == "llm_general"
    assert result.answered_by is not None
