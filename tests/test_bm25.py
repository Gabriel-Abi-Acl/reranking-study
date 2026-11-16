"""Contrato do BM25 Okapi: frequência, tamanho, IDF e escore numérico estável.

O IDF é o de Lucene, sempre >= 0. Reindexar substitui o corpus em vez de somar df.
"""

import pytest

from reranking_study.retrievers import Bm25Retriever
from reranking_study.types import Document

_CATS = [
    Document("d1", "gato gato"),
    Document("d2", "gato cachorro"),
    Document("d3", "cachorro cachorro cachorro"),
]


def _scores(documents: list[Document], query: str) -> dict[str, float]:
    retriever = Bm25Retriever()
    retriever.index(documents)
    return {hit.doc_id: hit.score for hit in retriever.retrieve(query, k=len(documents))}


def test_bm25_matches_lucene_idf_on_cat_corpus() -> None:
    scores = _scores(_CATS, "gato")
    assert scores["d1"] == pytest.approx(0.7037487496727592)
    assert scores["d2"] == pytest.approx(0.5022939549191068)
    assert scores["d3"] == 0.0
    assert scores["d1"] > scores["d2"] > scores["d3"]


def test_repeated_query_term_is_counted_once() -> None:
    assert _scores(_CATS, "gato gato") == _scores(_CATS, "gato")


def test_higher_term_frequency_scores_higher_at_the_same_length() -> None:
    documents = [
        Document("one", "termo outro outro outro"),
        Document("many", "termo termo termo outro"),
    ]
    scores = _scores(documents, "termo")
    assert scores["many"] > scores["one"]


def test_longer_document_scores_lower_at_the_same_term_frequency() -> None:
    documents = [
        Document("short", "termo outro"),
        Document("long", "termo outro outro outro outro"),
    ]
    scores = _scores(documents, "termo")
    assert scores["short"] > scores["long"]


def test_rarer_term_scores_higher_than_a_term_in_every_document() -> None:
    documents = [
        Document("d1", "raro comum"),
        Document("d2", "comum extra"),
        Document("d3", "comum nada"),
    ]
    rare = _scores(documents, "raro")["d1"]
    common = _scores(documents, "comum")["d1"]
    assert rare == pytest.approx(0.9808292530117263)
    assert common == pytest.approx(0.13353139262452257)
    assert rare > common > 0.0


def test_reindex_replaces_documents_and_drops_old_terms() -> None:
    retriever = Bm25Retriever()
    retriever.index([Document("d", "gato")])
    retriever.index([Document("d", "cachorro"), Document("a", "gato")])
    hits = {hit.doc_id: hit for hit in retriever.retrieve("gato", k=5)}
    assert hits["d"].text == "cachorro"
    assert hits["d"].score == 0.0
    assert hits["a"].score > 0.0


def test_empty_corpus_zero_k_and_invalid_parameters() -> None:
    retriever = Bm25Retriever()
    assert retriever.retrieve("gato", k=3) == []
    retriever.index(_CATS)
    assert retriever.retrieve("gato", k=0) == []
    with pytest.raises(ValueError):
        retriever.retrieve("gato", k=-1)
    with pytest.raises(ValueError):
        Bm25Retriever(k1=-0.1)
    with pytest.raises(ValueError):
        Bm25Retriever(b=1.1)
