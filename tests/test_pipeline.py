"""Contrato do pipeline: primeiro estágio, rerank opcional só no top-N, latência por estágio.

Os dobles abaixo fixam a ordem. O placeholder só garante que o gabarito aponta para ids reais.
"""

from pathlib import Path

import pytest

from reranking_study.corpus import DOCUMENTS, QUERIES
from reranking_study.pipeline import RetrievalPipeline
from reranking_study.rerankers import FakeReranker
from reranking_study.retrievers import Bm25Retriever, DenseRetriever
from reranking_study.types import Document, Hit


class _OrderRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.last_k: int | None = None

    def index(self, documents: list[Document]) -> None:
        return None

    def retrieve(self, query: str, k: int) -> list[Hit]:
        self.last_k = k
        return list(self.hits[:k])


class _SpyReranker:
    def __init__(self) -> None:
        self.seen: list[str] | None = None

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        self.seen = [hit.doc_id for hit in hits]
        return FakeReranker().rerank(query, hits, top_n)


def test_pipeline_reranks_only_the_fetched_top_n_and_records_latency() -> None:
    retriever = _OrderRetriever(
        [
            Hit("d1", 0.9, "xx"),
            Hit("d2", 0.2, "alpha beta"),
            Hit("d3", 0.1, "alpha beta extra"),
        ]
    )
    spy = _SpyReranker()
    pipeline = RetrievalPipeline(retriever, spy, rerank_top_n=2)
    result = pipeline.run("alpha beta", k=1)
    assert retriever.last_k == 2
    assert spy.seen == ["d1", "d2"]
    assert result.ids == ["d2"]
    assert result.scores == [2.0]
    assert set(result.stage_latency_ms) == {"retrieve", "rerank"}
    assert all(value >= 0.0 for value in result.stage_latency_ms.values())


def test_pipeline_appends_unreranked_tail_when_k_exceeds_top_n() -> None:
    retriever = _OrderRetriever(
        [
            Hit("d1", 0.9, "xx"),
            Hit("d2", 0.2, "alpha"),
            Hit("d3", 0.1, "tail"),
        ]
    )
    pipeline = RetrievalPipeline(retriever, FakeReranker(), rerank_top_n=2)
    result = pipeline.run("alpha", k=3)
    assert result.ids == ["d2", "d1", "d3"]


def test_pipeline_without_reranker_keeps_first_stage_order() -> None:
    retriever = _OrderRetriever([Hit("d1", 0.4, "aa"), Hit("d2", 0.2, "bb")])
    pipeline = RetrievalPipeline(retriever, reranker=None)
    result = pipeline.run("aa", k=1)
    assert result.ids == ["d1"]
    assert result.scores == [0.4]
    assert set(result.stage_latency_ms) == {"retrieve"}
    assert retriever.last_k == 1


def test_pipeline_with_zero_top_n_does_not_call_the_reranker() -> None:
    retriever = _OrderRetriever([Hit("d1", 0.4, "alpha")])

    class _Boom:
        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            raise AssertionError("rerank should not run when top-N is 0")

    pipeline = RetrievalPipeline(retriever, _Boom(), rerank_top_n=0)
    result = pipeline.run("alpha", k=1)
    assert result.ids == ["d1"]
    assert "rerank" not in result.stage_latency_ms
    with pytest.raises(ValueError):
        RetrievalPipeline(retriever, FakeReranker(), rerank_top_n=-1)
    with pytest.raises(ValueError):
        pipeline.run("alpha", k=-1)


def test_bm25_and_dense_pipelines_run_on_the_placeholder_corpus() -> None:
    bm25 = Bm25Retriever()
    bm25.index(DOCUMENTS)
    dense = DenseRetriever()
    dense.index(DOCUMENTS)
    bm25_result = RetrievalPipeline(bm25, FakeReranker(), rerank_top_n=3).run(QUERIES[0].text, k=2)
    dense_result = RetrievalPipeline(dense).run(QUERIES[0].text, k=2)
    known = {document.doc_id for document in DOCUMENTS}
    assert set(bm25_result.ids) <= known
    assert len(bm25_result.ids) == 2
    assert bm25.retrieve(QUERIES[0].text, k=1)[0].doc_id == "doc-bm25"
    dense_ids = [hit.doc_id for hit in dense.retrieve(QUERIES[0].text, k=2)]
    assert dense_result.ids == dense_ids
    assert len(dense_result.ids) == 2
    again = RetrievalPipeline(dense).run(QUERIES[0].text, k=2)
    assert again.ids == dense_result.ids
    assert again.scores == dense_result.scores


def test_placeholder_labels_point_at_real_documents() -> None:
    known = {document.doc_id for document in DOCUMENTS}
    assert known
    assert QUERIES
    for query in QUERIES:
        assert query.relevance
        assert set(query.relevance) <= known
        assert all(grade > 0 for grade in query.relevance.values())


def test_readme_skeleton_and_ci_pin_and_empty_env_example() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for heading in ("Objetivo", "Hipótese", "Metodologia planejada", "Como rodar", "Roadmap"):
        assert heading in readme
    assert "PR2" in readme and "PR3" in readme and "PR4" in readme
    assert not any(
        "\U0001f300" <= char <= "\U0001faff" or "\u2600" <= char <= "\u27bf" for char in readme
    )

    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@v10.3.3" in workflow
    assert 'version: "0.13.8"' in workflow
    assert "ruff check" in workflow
    assert "pytest" in workflow

    for line in Path(".env.example").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assert key.strip()
        assert value.strip() == ""
