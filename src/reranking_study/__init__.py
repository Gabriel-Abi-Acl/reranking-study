"""Estudo de pipelines de recuperação para RAG: denso, BM25, RRF e rerank.

Este pacote entrega o contrato, o conjunto v1 e os dobles determinísticos.
Número de experimento fica fora daqui.
"""

from reranking_study.corpus import DOCUMENTS, QUERIES
from reranking_study.eval_data import (
    ALLOWED_GRADES,
    EVAL_CATEGORIES,
    EvalPassage,
    EvalQuery,
    load_eval_corpus,
    load_eval_queries,
)
from reranking_study.metrics import mrr, ndcg_at_k, recall_at_k
from reranking_study.pipeline import RetrievalPipeline
from reranking_study.rerankers import CrossEncoderReranker, FakeReranker, LlmReranker
from reranking_study.retrievers import (
    Bm25Retriever,
    DenseRetriever,
    HashEmbedder,
    HybridRetriever,
    reciprocal_rank_fusion,
)
from reranking_study.types import Document, Hit, LabeledQuery, PipelineResult

__all__ = [
    "ALLOWED_GRADES",
    "DOCUMENTS",
    "EVAL_CATEGORIES",
    "QUERIES",
    "Bm25Retriever",
    "CrossEncoderReranker",
    "DenseRetriever",
    "Document",
    "EvalPassage",
    "EvalQuery",
    "FakeReranker",
    "HashEmbedder",
    "Hit",
    "HybridRetriever",
    "LabeledQuery",
    "LlmReranker",
    "PipelineResult",
    "RetrievalPipeline",
    "load_eval_corpus",
    "load_eval_queries",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank_fusion",
]
