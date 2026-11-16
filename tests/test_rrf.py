"""Contrato do RRF: soma 1 / (k + posição), sem usar a escala dos escores de origem.

Id repetido na mesma lista conta uma vez. O híbrido tem de reproduzir essa fusão.
"""

import pytest

from reranking_study.retrievers import HybridRetriever, reciprocal_rank_fusion
from reranking_study.types import Document


def test_rrf_sums_reciprocal_ranks_and_orders_by_score_then_id() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], rrf_k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "d", "c"]


def test_rrf_keeps_the_first_occurrence_inside_one_list() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], rrf_k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)


def test_rrf_tie_breaks_by_document_id() -> None:
    fused = reciprocal_rank_fusion([["b"], ["a"]], rrf_k=60)
    assert [doc_id for doc_id, _score in fused] == ["a", "b"]


def test_rrf_rejects_negative_k_and_ignores_empty_rankings() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], rrf_k=-1)


def test_hybrid_scores_match_fusion_of_the_two_rankings() -> None:
    documents = [
        Document("d1", "alfa beta beta"),
        Document("d2", "beta"),
        Document("d3", "gama delta epsilon"),
    ]
    hybrid = HybridRetriever(rrf_k=60)
    hybrid.index(documents)
    query = "beta"
    hits = hybrid.retrieve(query, k=3)
    dense_ids = [hit.doc_id for hit in hybrid.dense.retrieve(query, k=3)]
    bm25_ids = [hit.doc_id for hit in hybrid.bm25.retrieve(query, k=3)]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids], rrf_k=60)
    assert [hit.doc_id for hit in hits] == [doc_id for doc_id, _score in fused]
    for hit, (_doc_id, score) in zip(hits, fused, strict=True):
        assert hit.score == pytest.approx(score)
    again = hybrid.retrieve(query, k=3)
    assert [(hit.doc_id, hit.score) for hit in again] == [(hit.doc_id, hit.score) for hit in hits]


def test_hybrid_candidate_k_limits_each_list_before_fusion() -> None:
    documents = [
        Document("d1", "alfa beta beta"),
        Document("d2", "beta"),
        Document("d3", "gama"),
    ]
    hybrid = HybridRetriever(candidate_k=1)
    hybrid.index(documents)
    query = "beta"
    hits = hybrid.retrieve(query, k=5)
    allowed = {
        hybrid.dense.retrieve(query, k=1)[0].doc_id,
        hybrid.bm25.retrieve(query, k=1)[0].doc_id,
    }
    assert {hit.doc_id for hit in hits} <= allowed
    assert hits
