"""Pipeline: primeiro estágio e, se houver reranker, reordenação só do top-N.

Registra os ids, os escores da etapa final e a latência de cada estágio em milissegundos.
"""

import time

from reranking_study.rerankers import Reranker
from reranking_study.retrievers import Retriever
from reranking_study.types import Hit, PipelineResult

DEFAULT_RERANK_TOP_N = 10


class RetrievalPipeline:
    """Executa retrieve e, opcionalmente, rerank. k é o tamanho da lista devolvida.

    O reranker só vê os rerank_top_n primeiros hits. Se k passa desse N, o restante
    entra na ordem original do primeiro estágio, depois do bloco reranqueado.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        *,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> None:
        if rerank_top_n < 0:
            raise ValueError("rerank_top_n must be >= 0")
        self.retriever = retriever
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n

    def run(self, query: str, k: int) -> PipelineResult:
        if k < 0:
            raise ValueError("k must be >= 0")
        fetch_k = max(k, self.rerank_top_n) if self.reranker is not None else k
        started = time.perf_counter()
        first_hits = self.retriever.retrieve(query, fetch_k)
        latency: dict[str, float] = {"retrieve": _elapsed_ms(started)}
        # Corte em k antes do rerank. O harness compara o nDCG contra esta lista.
        first_stage = tuple(first_hits[:k])
        hits = list(first_stage)
        if self.reranker is not None and self.rerank_top_n > 0:
            pool = first_hits[: self.rerank_top_n]
            rerank_started = time.perf_counter()
            reranked = self.reranker.rerank(query, pool, len(pool))
            latency["rerank"] = _elapsed_ms(rerank_started)
            hits = _merge_reranked(first_hits, reranked, k)
        return PipelineResult(
            query=query,
            hits=tuple(hits),
            stage_latency_ms=dict(latency),
            first_stage_hits=first_stage,
        )


def _merge_reranked(first_hits: list[Hit], reranked: list[Hit], k: int) -> list[Hit]:
    if k <= len(reranked):
        return reranked[:k]
    seen = {hit.doc_id for hit in reranked}
    tail = [hit for hit in first_hits if hit.doc_id not in seen]
    return [*reranked, *tail][:k]


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
