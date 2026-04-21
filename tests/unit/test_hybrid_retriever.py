from app.pipeline.query_understander import understand
from app.retrieval.ecu_knowledge import ECUKnowledgeResult
from app.retrieval.hybrid_retriever import retrieve_hybrid
from app.utils.contracts import RetrievalResult, RetrievalStatus


def _not_found() -> RetrievalResult:
    return RetrievalResult(status=RetrievalStatus.NOT_FOUND)


def test_db_hit_returns_db(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.retrieval.hybrid_retriever.search",
        lambda *_, **__: RetrievalResult(status=RetrievalStatus.OK, canonical_name="Robotics Lab", confidence=0.95),
    )

    result = retrieve_hybrid(understand("where is the robotics lab"))

    assert result.answered_by == "db"
    assert result.db_result is not None


def test_db_ambiguous_returns_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.retrieval.hybrid_retriever.search",
        lambda *_, **__: RetrievalResult(status=RetrievalStatus.AMBIGUOUS, candidates=["A", "B"]),
    )

    result = retrieve_hybrid(understand("where is the lab"))

    assert result.answered_by == "clarification"


def test_db_miss_ecu_hit(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.hybrid_retriever.search", lambda *_, **__: _not_found())
    monkeypatch.setattr(
        "app.retrieval.hybrid_retriever.search_ecu_knowledge",
        lambda *_: ECUKnowledgeResult(found=True, title="Library", content="ECU Library"),
    )

    result = retrieve_hybrid(understand("library"))

    assert result.answered_by == "ecu_web"
    assert result.ecu_result is not None


def test_db_miss_ecu_miss_general(monkeypatch) -> None:
    monkeypatch.setattr("app.retrieval.hybrid_retriever.search", lambda *_, **__: _not_found())
    monkeypatch.setattr(
        "app.retrieval.hybrid_retriever.search_ecu_knowledge",
        lambda *_: ECUKnowledgeResult(found=False),
    )

    result = retrieve_hybrid(understand("unknown thing"))

    assert result.answered_by == "llm_general"


def test_fallbacks_tried_in_order(monkeypatch) -> None:
    calls = []

    def fake_search(query, *_, **__):
        calls.append(query)
        if query == "robotics lab":
            return RetrievalResult(status=RetrievalStatus.OK, canonical_name="Robotics Lab", confidence=0.95)
        return _not_found()

    monkeypatch.setattr("app.retrieval.hybrid_retriever.search", fake_search)

    result = retrieve_hybrid(understand("tell me about the robotics lab", router_entity="wrong", router_confidence=0.9))

    assert result.answered_by == "db"
    assert calls[:2] == ["wrong", "robotics lab"]
