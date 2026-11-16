"""Contrato das métricas: recall e MRR tratam grau > 0 como relevante; nDCG usa o grau.

Id duplicado na lista recuperada não aumenta recall nem adianta a posição do MRR.
"""

import pytest

from reranking_study.metrics import mrr, ndcg_at_k, recall_at_k

_RELEVANCE = {"d1": 3, "d2": 2, "d3": 0, "d4": 1}


def test_recall_at_k_uses_positive_grades_and_ignores_duplicate_ids() -> None:
    retrieved = ["d1", "d1", "d3", "d2"]
    assert recall_at_k(retrieved, _RELEVANCE, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, _RELEVANCE, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(["d3"], _RELEVANCE, k=1) == 0.0
    assert recall_at_k(["d1"], {"d1": 0}, k=1) == 0.0
    assert recall_at_k(["d1"], _RELEVANCE, k=0) == 0.0


def test_mrr_is_the_reciprocal_rank_of_the_first_relevant_hit() -> None:
    assert mrr(["d3", "d1"], _RELEVANCE) == pytest.approx(0.5)
    assert mrr(["d3", "d3", "d4"], _RELEVANCE) == pytest.approx(1 / 2)
    assert mrr(["d3"], _RELEVANCE) == 0.0
    assert mrr(["d1"], _RELEVANCE) == 1.0


def test_ndcg_at_k_uses_graded_gain_and_is_one_for_the_ideal_order() -> None:
    assert ndcg_at_k(["d1", "d2", "d3"], _RELEVANCE, k=3) == pytest.approx(0.9467676761267002)
    assert ndcg_at_k(["d1", "d2", "d4"], _RELEVANCE, k=3) == pytest.approx(1.0)
    assert ndcg_at_k(["d3"], {"d3": 0, "d9": 0}, k=1) == 0.0
    assert ndcg_at_k(["d1"], _RELEVANCE, k=0) == 0.0


def test_metrics_reject_negative_k_and_negative_grades() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["d1"], _RELEVANCE, k=-1)
    with pytest.raises(ValueError):
        ndcg_at_k(["d1"], {"d1": -1}, k=1)
    with pytest.raises(ValueError):
        mrr(["d1"], {"d1": True})
