"""Métricas de ranking por consulta: recall@k, MRR e nDCG@k com relevância graduada.

Funções puras sobre ids e gabarito. Não agregam corpus nem inventam grau ausente: ausência vale 0.
"""

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Fração dos documentos com grau > 0 que aparecem até a posição k.

    Sem relevante no gabarito, ou com k = 0, o valor é 0 para não produzir NaN.
    """

    _validate_k(k)
    _validate_relevance(relevance)
    relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if k == 0 or not relevant:
        return 0.0
    hit = {doc_id for doc_id in _dedupe(retrieved_ids)[:k] if doc_id in relevant}
    return len(hit) / len(relevant)


def mrr(retrieved_ids: Sequence[str], relevance: Mapping[str, int]) -> float:
    """Inverso da posição do primeiro documento com grau > 0. Zero se nenhum aparece.

    O corte @k, se existir, é responsabilidade de quem passa a lista já cortada.
    """

    _validate_relevance(relevance)
    for rank, doc_id in enumerate(_dedupe(retrieved_ids), start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """nDCG@k. Ganho é 2^grau - 1 e o desconto é log2(posição + 1).

    IDCG usa o gabarito em ordem decrescente de grau. IDCG 0 devolve 0, não uma divisão por zero.
    """

    _validate_k(k)
    _validate_relevance(relevance)
    if k == 0:
        return 0.0
    ideal_ids = sorted(relevance, key=lambda doc_id: (-relevance[doc_id], doc_id))
    idcg = _dcg(ideal_ids, relevance, k)
    if idcg == 0.0:
        return 0.0
    return _dcg(_dedupe(retrieved_ids), relevance, k) / idcg


def _dcg(ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    total = 0.0
    for rank, doc_id in enumerate(ids[:k], start=1):
        gain = (2 ** relevance.get(doc_id, 0)) - 1
        total += gain / math.log2(rank + 1)
    return total


def _dedupe(ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ids))


def _validate_k(k: int) -> None:
    if k < 0:
        raise ValueError("k must be >= 0")


def _validate_relevance(relevance: Mapping[str, int]) -> None:
    for grade in relevance.values():
        if isinstance(grade, bool) or not isinstance(grade, int | float):
            raise ValueError("relevance grades must be numbers")
        if grade < 0:
            raise ValueError("relevance grades must be >= 0")
