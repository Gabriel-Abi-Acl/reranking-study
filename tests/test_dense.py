"""Contrato do embedder hash: o mesmo texto gera o mesmo vetor unitário.

A recuperação densa só promete ordenação estável por cosseno, não semântica de modelo.
"""

import math

import pytest

from reranking_study.retrievers import DenseRetriever, HashEmbedder
from reranking_study.types import Document


def test_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashEmbedder(dim=32)
    first = embedder.embed("Recuperação densa por cosseno")
    second = embedder.embed("Recuperação densa por cosseno")
    assert first == second
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert embedder.embed("") == tuple(0.0 for _ in range(32))
    assert embedder.embed("Recuperação densa por cosseno") != embedder.embed("outro texto")
    with pytest.raises(ValueError):
        HashEmbedder(dim=0)


def test_dense_retrieval_is_sorted_and_replaces_the_index() -> None:
    retriever = DenseRetriever(HashEmbedder(dim=32))
    documents = [
        Document("b", "beta beta"),
        Document("a", "alfa beta"),
        Document("c", "gama"),
    ]
    retriever.index(documents)
    first = retriever.retrieve("beta", k=5)
    second = retriever.retrieve("beta", k=5)
    assert [(hit.doc_id, hit.score) for hit in first] == [(hit.doc_id, hit.score) for hit in second]
    assert [hit.score for hit in first] == sorted((hit.score for hit in first), reverse=True)
    assert len(first) == 3
    retriever.index([Document("z", "zeta")])
    hits = retriever.retrieve("beta", k=5)
    assert [hit.doc_id for hit in hits] == ["z"]
