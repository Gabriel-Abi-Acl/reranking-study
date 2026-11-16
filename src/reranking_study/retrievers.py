"""Retrievers do primeiro estágio: denso com hash, BM25 puro e híbrido com RRF.

O hash é determinístico para o teste não depender de modelo. O BM25 não chama biblioteca externa.
"""

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from reranking_study.index import InMemoryIndex
from reranking_study.tokenize import tokenize
from reranking_study.types import Document, Hit

DEFAULT_RRF_K = 60
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_EMBED_DIM = 64


class Retriever(Protocol):
    """Contrato do primeiro estágio: indexa o corpus e devolve os top-k hits."""

    def index(self, documents: Sequence[Document]) -> None: ...

    def retrieve(self, query: str, k: int) -> list[Hit]: ...


class HashEmbedder:
    """Embedder falso: cada token soma +1 ou -1 num balde SHA-256 e o vetor é L2-normalizado.

    O mesmo texto produz o mesmo vetor. Não representa semântica de um modelo real.
    """

    def __init__(self, dim: int = DEFAULT_EMBED_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


class DenseRetriever:
    """Recuperação densa por cosseno. Com vetores normalizados o cosseno é o produto interno.

    index() substitui o corpus anterior em vez de acumular ids.
    """

    def __init__(self, embedder: HashEmbedder | None = None) -> None:
        self.embedder = embedder or HashEmbedder()
        self._index = InMemoryIndex()
        self._vectors: dict[str, tuple[float, ...]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def index(self, documents: Sequence[Document]) -> None:
        self._index = InMemoryIndex()
        self._vectors = {}
        self._index.upsert(documents)
        for document in self._index.documents():
            self._vectors[document.doc_id] = self.embedder.embed(document.text)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        _validate_k(k)
        if k == 0 or len(self._index) == 0:
            return []
        query_vector = self.embedder.embed(query)
        hits = [
            Hit(document.doc_id, _dot(query_vector, self._vectors[document.doc_id]), document.text)
            for document in self._index.documents()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:k]


class Bm25Retriever:
    """Okapi BM25 em Python puro, com IDF não negativo no estilo Lucene.

    IDF(t) = ln(1 + (N - df + 0.5) / (df + 0.5)). Cada termo distinto da consulta entra uma vez.
    k1 e b seguem os defaults clássicos 1.5 e 0.75.
    """

    def __init__(self, k1: float = DEFAULT_BM25_K1, b: float = DEFAULT_BM25_B) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("k1 must be >= 0 and b must be in [0, 1]")
        self.k1 = k1
        self.b = b
        self._index = InMemoryIndex()
        self._term_freq: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}
        self._doc_freq: dict[str, int] = {}
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._index)

    def index(self, documents: Sequence[Document]) -> None:
        self._index = InMemoryIndex()
        self._index.upsert(documents)
        self._term_freq = {}
        self._doc_len = {}
        self._doc_freq = {}
        total_len = 0
        for document in self._index.documents():
            counts: dict[str, int] = {}
            for token in tokenize(document.text):
                counts[token] = counts.get(token, 0) + 1
            self._term_freq[document.doc_id] = counts
            length = sum(counts.values())
            self._doc_len[document.doc_id] = length
            total_len += length
            for token in counts:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        self._avgdl = total_len / len(self._index) if len(self._index) else 0.0

    def retrieve(self, query: str, k: int) -> list[Hit]:
        _validate_k(k)
        if k == 0 or len(self._index) == 0:
            return []
        query_terms = list(dict.fromkeys(tokenize(query)))
        hits = [
            Hit(document.doc_id, self._score(document.doc_id, query_terms), document.text)
            for document in self._index.documents()
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return hits[:k]

    def _score(self, doc_id: str, query_terms: Sequence[str]) -> float:
        if self._avgdl == 0.0:
            return 0.0
        length = self._doc_len[doc_id]
        counts = self._term_freq[doc_id]
        total = 0.0
        for term in query_terms:
            df = self._doc_freq.get(term)
            if not df:
                continue
            freq = counts.get(term, 0)
            idf = math.log(1.0 + (len(self._index) - df + 0.5) / (df + 0.5))
            norm = 1.0 - self.b + self.b * (length / self._avgdl)
            denom = freq + self.k1 * norm
            total += idf * (freq * (self.k1 + 1.0) / denom)
        return total


class HybridRetriever:
    """Funde o ranking denso e o BM25 com Reciprocal Rank Fusion.

    A fusão usa a posição, não a escala do escore. Por default cada lista vê o corpus inteiro.
    """

    def __init__(
        self,
        dense: DenseRetriever | None = None,
        bm25: Bm25Retriever | None = None,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int | None = None,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if candidate_k is not None and candidate_k < 0:
            raise ValueError("candidate_k must be >= 0")
        self.dense = dense or DenseRetriever()
        self.bm25 = bm25 or Bm25Retriever()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def index(self, documents: Sequence[Document]) -> None:
        self.dense.index(documents)
        self.bm25.index(documents)

    def retrieve(self, query: str, k: int) -> list[Hit]:
        _validate_k(k)
        if k == 0:
            return []
        corpus_size = max(len(self.dense), len(self.bm25))
        fetch_k = self.candidate_k if self.candidate_k is not None else max(k, corpus_size)
        dense_hits = self.dense.retrieve(query, fetch_k)
        bm25_hits = self.bm25.retrieve(query, fetch_k)
        fused = reciprocal_rank_fusion(
            [[hit.doc_id for hit in dense_hits], [hit.doc_id for hit in bm25_hits]],
            rrf_k=self.rrf_k,
        )
        text_by_id = {hit.doc_id: hit.text for hit in dense_hits}
        text_by_id.update({hit.doc_id: hit.text for hit in bm25_hits})
        return [Hit(doc_id, score, text_by_id.get(doc_id, "")) for doc_id, score in fused[:k]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[tuple[str, float]]:
    """Soma 1 / (rrf_k + posição) de cada lista. Posição começa em 1.

    Repetido na mesma lista sai antes da posição ser atribuída. Empate de escore desempata pelo id.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")
    scores: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        unique: list[str] = []
        for doc_id in ranking:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique.append(doc_id)
        for rank, doc_id in enumerate(unique, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _validate_k(k: int) -> None:
    if k < 0:
        raise ValueError("k must be >= 0")
