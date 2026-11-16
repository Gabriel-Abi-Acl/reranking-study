"""Tipos compartilhados entre índice, retrievers, rerankers e métricas.

O id do documento é a chave do gabarito; escore e texto acompanham o hit para o rerank.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Texto indexável. O id é estável entre o corpus e os rótulos de relevância."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    """Documento recuperado, com o escore da etapa que o produziu.

    O texto viaja com o hit para o reranker não depender de outro índice.
    """

    doc_id: str
    score: float
    text: str


@dataclass(frozen=True)
class LabeledQuery:
    """Consulta de avaliação. relevance mapeia doc_id para grau (>= 0).

    Grau 0 é irrelevante. Graus maiores são relevância graduada para o nDCG.
    """

    query_id: str
    text: str
    relevance: Mapping[str, int]


@dataclass(frozen=True)
class PipelineResult:
    """Saída de uma consulta: ranking final, primeiro estágio e latência em milissegundos.

    hits está na ordem devolvida ao avaliador. first_stage_hits é esse corte antes do
    rerank, para o lift usar a mesma busca. stage_latency_ms não inclui I/O de modelo
    que tenha ocorrido fora do método chamado.
    """

    query: str
    hits: tuple[Hit, ...]
    stage_latency_ms: Mapping[str, float]
    first_stage_hits: tuple[Hit, ...] = ()

    @property
    def ids(self) -> list[str]:
        return [hit.doc_id for hit in self.hits]

    @property
    def scores(self) -> list[float]:
        return [hit.score for hit in self.hits]
