"""Corpus sintético mínimo em português e consultas com gabarito.

Fixture do harness. O conjunto versionado da comparação fica em data/eval/v1.
"""

from reranking_study.types import Document, LabeledQuery

DOCUMENTS: tuple[Document, ...] = (
    Document(
        "doc-denso",
        "A recuperação densa representa consulta e documento como vetores e ordena pelo cosseno.",
    ),
    Document(
        "doc-bm25",
        "BM25 é um ranking lexical pela frequência do termo, a raridade e o tamanho.",
    ),
    Document(
        "doc-rrf",
        "A fusão reciprocal rank fusion soma o inverso da posição nas listas densa e lexical.",
    ),
    Document(
        "doc-reranker",
        "O reranker cross-encoder lê consulta e documento juntos e devolve um escore.",
    ),
    Document(
        "doc-indice",
        "O índice em memória guarda os textos e as estatísticas das consultas de teste.",
    ),
)

QUERIES: tuple[LabeledQuery, ...] = (
    LabeledQuery(
        "q-bm25",
        "BM25",
        {"doc-bm25": 2, "doc-rrf": 1},
    ),
    LabeledQuery(
        "q-rerank",
        "reranqueamento com cross-encoder",
        {"doc-reranker": 2, "doc-denso": 1},
    ),
    LabeledQuery(
        "q-hibrido",
        "fusão de ranking denso e lexical",
        {"doc-rrf": 2, "doc-denso": 1, "doc-bm25": 1},
    ),
)
