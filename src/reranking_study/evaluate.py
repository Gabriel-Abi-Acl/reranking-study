"""Harness offline: roda os pipelines no v1 e grava JSONL mais um agregado.

O embedder e o reranker padrão são determinísticos. Número desse caminho
confere o encanamento; não é achado. Componente real só segue com
--allow-live, modelo e chave, e mesmo assim este módulo não imprime métrica.
"""

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from reranking_study.eval_data import (
    EVAL_CATEGORIES,
    EvalPassage,
    EvalQuery,
    load_eval_corpus,
    load_eval_queries,
)
from reranking_study.metrics import mrr, ndcg_at_k, recall_at_k
from reranking_study.pipeline import DEFAULT_RERANK_TOP_N, RetrievalPipeline
from reranking_study.rerankers import CrossEncoderReranker, FakeReranker, LlmReranker, Reranker
from reranking_study.retrievers import (
    Bm25Retriever,
    DenseRetriever,
    HashEmbedder,
    HybridRetriever,
    Retriever,
)
from reranking_study.types import Document, Hit, PipelineResult

SCHEMA_VERSION = 1
RETRIEVAL_K = 10
RECALL_KS = (1, 3, 5, 10)
NDCG_KS = (5, 10)
MRR_K = 10
RECORDS_FILENAME = "records.jsonl"
SUMMARY_FILENAME = "summary.json"
DATASET_NAME = "v1"
DEFAULT_EVAL_DIR = "data/eval/v1"
DEFAULT_OUTPUT_DIR = "results/eval"

FIRST_STAGES = ("dense", "bm25", "hybrid_rrf")
PIPELINES = FIRST_STAGES + tuple(f"{name}_rerank" for name in FIRST_STAGES)
EMBEDDERS = ("fake", "openai")
RERANKERS = ("fake", "cross-encoder", "llm")
CATEGORY_ORDER = ("lexical", "paraphrase", "distractor", "multi")

FAKE_DISCLAIMER = (
    "Números com embedder e reranker determinísticos conferem o pipeline. "
    "Não são achados do estudo."
)
LIVE_DISCLAIMER = (
    "Execução com componente real. Este arquivo não é o relatório publicado do estudo."
)

RECORD_KEYS = (
    "schema_version",
    "embedder",
    "reranker",
    "pipeline",
    "first_stage",
    "rerank",
    "rerank_top_n",
    "query_id",
    "category",
    "retrieved_ids",
    "first_stage_ids",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_10_lift",
    "stage_latency_ms",
    "reranker_calls",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "error",
)

_METRIC_KEYS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "mrr_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_10_lift",
)

AGGREGATE_KEYS = (
    "embedder",
    "reranker",
    "pipeline",
    "first_stage",
    "rerank",
    "rerank_top_n",
    "query_count",
    "error_count",
    *_METRIC_KEYS,
    "stage_latency_ms",
    "reranker_calls",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "by_category",
)

CATEGORY_KEYS = (
    "category",
    "query_count",
    "error_count",
    *_METRIC_KEYS,
    "stage_latency_ms",
    "reranker_calls",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
)

SUMMARY_KEYS = (
    "schema_version",
    "dataset",
    "eval_dir",
    "embedder",
    "reranker",
    "rerank_top_n",
    "retrieval_k",
    "recall_ks",
    "ndcg_ks",
    "mrr_k",
    "passage_count",
    "query_count",
    "row_count",
    "fake_components",
    "disclaimer",
    "pipelines",
)

_LATENCY_STAGES = ("retrieve", "rerank")

if set(CATEGORY_ORDER) != EVAL_CATEGORIES:
    raise RuntimeError("CATEGORY_ORDER divergiu das categorias do v1.")


class ConfigError(Exception):
    """Argumento ou arquivo que impede a corrida antes de gravar saída.

    Falha de uma consulta não usa esta classe: a linha guarda o erro e a média o ignora.
    """


class Embedder(Protocol):
    """Vetor de um texto. O hash não sai da máquina; o cliente OpenAI só nasce na chamada."""

    def embed(self, text: str) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class ModelUsage:
    """Uso informado pelo componente. Campo ausente fica null; este tipo não inventa preço."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class EvalRecord:
    """Uma consulta em um pipeline. Métrica null com error preenchido fica fora da média."""

    embedder: str
    reranker: str
    pipeline: str
    first_stage: str
    rerank: bool
    rerank_top_n: int | None
    query_id: str
    category: str
    retrieved_ids: tuple[str, ...] | None
    first_stage_ids: tuple[str, ...] | None
    recall_at_1: float | None
    recall_at_3: float | None
    recall_at_5: float | None
    recall_at_10: float | None
    mrr_at_10: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    ndcg_at_10_lift: float | None
    stage_latency_ms: dict[str, float | None] | None
    reranker_calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    error: str | None


@dataclass(frozen=True)
class _Identity:
    embedder: str
    reranker: str
    pipeline: str
    first_stage: str
    rerank: bool
    rerank_top_n: int | None


class _CountingReranker:
    """Conta chamadas do reranker e copia o uso se o componente o publicou em last_usage.

    A contagem sobe antes do retorno: falha de rede ainda entra no proxy de custo.
    """

    def __init__(self, inner: Reranker) -> None:
        self._inner = inner
        self.calls = 0
        self.usage: ModelUsage | None = None

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        self.calls += 1
        try:
            return self._inner.rerank(query, hits, top_n)
        finally:
            self.usage = _usage_of(self._inner)


class OpenAiEmbedder:
    """Embedder da API OpenAI. O cliente só nasce na primeira chamada de embed.

    Sem o extra llm, a chamada levanta ImportError em vez de devolver vetor inventado.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client: object | None = None

    def embed(self, text: str) -> tuple[float, ...]:
        client = self._get_client()
        response = client.embeddings.create(model=self.model, input=text)
        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("OpenAI embeddings response did not contain a vector")
        raw = getattr(data[0], "embedding", None)
        if not raw:
            raise RuntimeError("OpenAI embeddings response did not contain a vector")
        return _l2_normalize([float(value) for value in raw])

    def _get_client(self) -> object:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("OpenAiEmbedder requires the optional 'llm' extra") from exc
            kwargs: dict[str, str] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client


def expand_pipelines(name: str) -> tuple[str, ...]:
    """`all` devolve os seis pipelines na ordem estável do estudo."""

    normalized = name.strip().lower()
    if normalized == "all":
        return PIPELINES
    if normalized not in PIPELINES:
        options = ", ".join((*PIPELINES, "all"))
        raise ConfigError(f"pipeline inválido: {name}. Use: {options}.")
    return (normalized,)


def split_pipeline(name: str) -> tuple[str, bool]:
    """Separa o primeiro estágio da marca de rerank. Nome fora da lista é erro de configuração."""

    if name.endswith("_rerank"):
        first_stage = name[: -len("_rerank")]
        if first_stage not in FIRST_STAGES:
            raise ConfigError(f"pipeline inválido: {name}.")
        return first_stage, True
    if name not in FIRST_STAGES:
        raise ConfigError(f"pipeline inválido: {name}.")
    return name, False


def live_env_names(embedder: str, reranker: str) -> tuple[str, ...]:
    """Variáveis que precisam estar preenchidas. Default de código não conta como presença."""

    names: list[str] = []
    if embedder == "openai":
        names.extend(["OPENAI_API_KEY", "OPENAI_EMBEDDING_MODEL"])
    if reranker == "llm":
        names.extend(["OPENAI_API_KEY", "OPENAI_MODEL"])
    if reranker == "cross-encoder":
        names.append("CROSS_ENCODER_MODEL")
    return tuple(dict.fromkeys(names))


def ensure_live_allowed(embedder: str, reranker: str, *, allow_live: bool) -> None:
    """Recusa componente real sem flag ou sem variável. Sai com código 2 e não constrói cliente."""

    if embedder == "fake" and reranker == "fake":
        return
    if not allow_live:
        print(
            "Embedder ou reranker real faz chamada de rede ou download de modelo. "
            "Passe --allow-live para autorizar. Nenhuma chamada foi feita.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    missing = [name for name in live_env_names(embedder, reranker) if not _env(name)]
    if missing:
        joined = ", ".join(missing)
        print(
            f"Faltam variáveis no ambiente: {joined}. Nenhuma chamada foi feita.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def build_embedder(name: str, *, api_key: str, model: str, base_url: str) -> Embedder:
    """Hash para fake. OpenAI só é construído depois do portão; o cliente HTTP nasce em embed."""

    if name == "fake":
        return HashEmbedder()
    if name == "openai":
        return OpenAiEmbedder(api_key=api_key, model=model, base_url=base_url)
    options = ", ".join(EMBEDDERS)
    raise ConfigError(f"embedder inválido: {name}. Use: {options}.")


def build_reranker(name: str) -> Reranker:
    """Reranker falso, cross-encoder ou LLM. Os dois últimos importam a lib só na chamada."""

    if name == "fake":
        return FakeReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker(model_name=_env("CROSS_ENCODER_MODEL"))
    if name == "llm":
        return LlmReranker()
    options = ", ".join(RERANKERS)
    raise ConfigError(f"reranker inválido: {name}. Use: {options}.")


def make_retriever(first_stage: str, embedder: Embedder) -> Retriever:
    """Um retriever novo por pipeline, para o índice de um não vazar no seguinte."""

    if first_stage == "dense":
        return DenseRetriever(embedder)  # type: ignore[arg-type]
    if first_stage == "bm25":
        return Bm25Retriever()
    if first_stage == "hybrid_rrf":
        dense = DenseRetriever(embedder)  # type: ignore[arg-type]
        return HybridRetriever(dense, Bm25Retriever())
    raise ConfigError(f"primeiro estágio inválido: {first_stage}.")


def evaluate_pipeline(
    pipeline: str,
    documents: Sequence[Document],
    queries: Sequence[EvalQuery],
    *,
    retriever: Retriever,
    reranker: Reranker | None,
    rerank_top_n: int,
    embedder_name: str,
    reranker_name: str,
    secrets: Sequence[str] = (),
) -> list[EvalRecord]:
    """Indexa uma vez e devolve uma linha por consulta.

    Exceção vira error na linha. A métrica dessa linha fica null para a média não tratar
    falha como zero. Chamada de rerank que estourou ainda conta no proxy de custo.
    """

    identity = _Identity(
        embedder=embedder_name,
        reranker=reranker_name,
        pipeline=pipeline,
        first_stage=split_pipeline(pipeline)[0],
        rerank=reranker is not None,
        rerank_top_n=rerank_top_n if reranker is not None else None,
    )
    try:
        retriever.index(documents)
    except Exception as exc:
        message = _public_error(exc, secrets)
        return [_error_record(identity, query, message, calls=0, usage=None) for query in queries]

    counter = _CountingReranker(reranker) if reranker is not None else None
    runner = RetrievalPipeline(
        retriever,
        counter,
        rerank_top_n=rerank_top_n,
    )
    records: list[EvalRecord] = []
    for query in queries:
        if counter is not None:
            counter.calls = 0
            counter.usage = None
        try:
            result = runner.run(query.text, RETRIEVAL_K)
        except Exception as exc:
            usage = counter.usage if counter is not None else None
            records.append(
                _error_record(
                    identity,
                    query,
                    _public_error(exc, secrets),
                    calls=0 if counter is None else counter.calls,
                    usage=usage,
                )
            )
            continue
        records.append(_success_record(identity, query, result, counter))
    return records


def aggregate_records(records: Sequence[EvalRecord]) -> list[dict[str, object]]:
    """Uma entrada por pipeline. A média ignora erro e null, e conserva o zero.

    Lift sem rerank continua null. Token e custo só somam quando alguma linha os trouxe.
    """

    groups: dict[tuple[object, ...], list[EvalRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)
    blocks: list[dict[str, object]] = []
    for key in sorted(groups, key=_group_sort):
        group = groups[key]
        block = _metrics_block(group, category=None)
        block["by_category"] = _categories(group)
        if tuple(block) != AGGREGATE_KEYS:
            missing = set(AGGREGATE_KEYS) - set(block)
            raise RuntimeError(f"schema do pipeline divergiu: {sorted(missing)}.")
        blocks.append(block)
    return blocks


def build_summary(
    records: Sequence[EvalRecord],
    *,
    eval_dir: str,
    embedder: str,
    reranker: str,
    rerank_top_n: int,
    passage_count: int,
    query_count: int,
) -> dict[str, object]:
    """Agregado da corrida. O aviso distingue checagem com dobles de relatório do estudo."""

    fake = embedder == "fake" and reranker == "fake"
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_NAME,
        "eval_dir": eval_dir,
        "embedder": embedder,
        "reranker": reranker,
        "rerank_top_n": rerank_top_n,
        "retrieval_k": RETRIEVAL_K,
        "recall_ks": list(RECALL_KS),
        "ndcg_ks": list(NDCG_KS),
        "mrr_k": MRR_K,
        "passage_count": passage_count,
        "query_count": query_count,
        "row_count": len(records),
        "fake_components": fake,
        "disclaimer": FAKE_DISCLAIMER if fake else LIVE_DISCLAIMER,
        "pipelines": aggregate_records(records),
    }
    if tuple(summary) != SUMMARY_KEYS:
        missing = set(SUMMARY_KEYS) - set(summary)
        raise RuntimeError(f"schema do agregado divergiu: {sorted(missing)}.")
    return summary


def record_to_dict(record: EvalRecord) -> dict[str, object]:
    raw: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "embedder": record.embedder,
        "reranker": record.reranker,
        "pipeline": record.pipeline,
        "first_stage": record.first_stage,
        "rerank": record.rerank,
        "rerank_top_n": record.rerank_top_n,
        "query_id": record.query_id,
        "category": record.category,
        "retrieved_ids": None if record.retrieved_ids is None else list(record.retrieved_ids),
        "first_stage_ids": None if record.first_stage_ids is None else list(record.first_stage_ids),
        "recall_at_1": record.recall_at_1,
        "recall_at_3": record.recall_at_3,
        "recall_at_5": record.recall_at_5,
        "recall_at_10": record.recall_at_10,
        "mrr_at_10": record.mrr_at_10,
        "ndcg_at_5": record.ndcg_at_5,
        "ndcg_at_10": record.ndcg_at_10,
        "ndcg_at_10_lift": record.ndcg_at_10_lift,
        "stage_latency_ms": record.stage_latency_ms,
        "reranker_calls": record.reranker_calls,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "cost_usd": record.cost_usd,
        "error": record.error,
    }
    if tuple(raw) != RECORD_KEYS:
        missing = set(RECORD_KEYS) - set(raw)
        raise RuntimeError(f"schema do registro divergiu: {sorted(missing)}.")
    return raw


def eval_record_from_dict(raw: Mapping[str, object]) -> EvalRecord:
    missing = [key for key in RECORD_KEYS if key not in raw]
    if missing:
        names = ", ".join(missing)
        raise ConfigError(f"Registro sem campo: {names}.")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ConfigError("schema_version do registro não é a desta versão.")
    return EvalRecord(
        embedder=_text(raw["embedder"], "embedder"),
        reranker=_text(raw["reranker"], "reranker"),
        pipeline=_text(raw["pipeline"], "pipeline"),
        first_stage=_text(raw["first_stage"], "first_stage"),
        rerank=_required_bool(raw["rerank"], "rerank"),
        rerank_top_n=_optional_int(raw["rerank_top_n"], "rerank_top_n"),
        query_id=_text(raw["query_id"], "query_id"),
        category=_text(raw["category"], "category"),
        retrieved_ids=_ids(raw["retrieved_ids"], "retrieved_ids"),
        first_stage_ids=_ids(raw["first_stage_ids"], "first_stage_ids"),
        recall_at_1=_optional_float(raw["recall_at_1"], "recall_at_1"),
        recall_at_3=_optional_float(raw["recall_at_3"], "recall_at_3"),
        recall_at_5=_optional_float(raw["recall_at_5"], "recall_at_5"),
        recall_at_10=_optional_float(raw["recall_at_10"], "recall_at_10"),
        mrr_at_10=_optional_float(raw["mrr_at_10"], "mrr_at_10"),
        ndcg_at_5=_optional_float(raw["ndcg_at_5"], "ndcg_at_5"),
        ndcg_at_10=_optional_float(raw["ndcg_at_10"], "ndcg_at_10"),
        ndcg_at_10_lift=_optional_float(raw["ndcg_at_10_lift"], "ndcg_at_10_lift"),
        stage_latency_ms=_latency(raw["stage_latency_ms"]),
        reranker_calls=_required_int(raw["reranker_calls"], "reranker_calls"),
        prompt_tokens=_optional_int(raw["prompt_tokens"], "prompt_tokens"),
        completion_tokens=_optional_int(raw["completion_tokens"], "completion_tokens"),
        cost_usd=_optional_float(raw["cost_usd"], "cost_usd"),
        error=None if raw["error"] is None else _text(raw["error"], "error"),
    )


def write_evaluation(
    output: Path, records: Sequence[EvalRecord], summary: Mapping[str, object]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [record_to_dict(record) for record in records]
    (output / RECORDS_FILENAME).write_text(_dumps_jsonl(rows), encoding="utf-8")
    (output / SUMMARY_FILENAME).write_text(_dumps_json(summary), encoding="utf-8")


def passages_as_documents(passages: Sequence[EvalPassage]) -> tuple[Document, ...]:
    return tuple(Document(passage.doc_id, passage.text) for passage in passages)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate-retrieval",
        description=(
            "Avalia pipelines de recuperação no conjunto v1 e grava JSONL e um agregado. "
            "O padrão é o embedder hash e o reranker determinístico, só para conferir o pipeline."
        ),
    )
    parser.add_argument(
        "--pipeline",
        default="all",
        help="dense, bm25, hybrid_rrf, o mesmo nome com _rerank, ou all.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=DEFAULT_RERANK_TOP_N,
        help="Quantos hits do primeiro estágio o rerank reordena. Padrão: 10.",
    )
    parser.add_argument("--embedder", default="fake", help="fake ou openai. Padrão: fake.")
    parser.add_argument(
        "--reranker",
        default="fake",
        help="fake, cross-encoder ou llm. Padrão: fake.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Autoriza rede ou download para embedder ou reranker real. Sem isto o comando sai 2.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Diretório de records.jsonl e summary.json. Padrão: results/eval.",
    )
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Diretório do conjunto. Padrão: data/eval/v1 ao lado do código.",
    )
    namespace = parser.parse_args(None if argv is None else list(argv))
    namespace.output = Path(namespace.output)
    namespace.eval_dir = None if namespace.eval_dir is None else Path(namespace.eval_dir)
    return namespace


def main(argv: Sequence[str] | None = None) -> None:
    try:
        records, output = execute(argv)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Registros: {len(records)}. Diretório: {output.as_posix()}")


def execute(argv: Sequence[str] | None = None) -> tuple[list[EvalRecord], Path]:
    """Roda a avaliação. O portão de componente real dispara SystemExit 2 antes de gravar."""

    args = parse_args(argv)
    pipelines = expand_pipelines(args.pipeline)
    embedder_name = _choice(args.embedder, EMBEDDERS, "embedder")
    reranker_name = _choice(args.reranker, RERANKERS, "reranker")
    if args.rerank_top_n < 1:
        raise ConfigError("--rerank-top-n precisa ser >= 1.")
    ensure_live_allowed(embedder_name, reranker_name, allow_live=args.allow_live)
    directory = args.eval_dir
    try:
        passages = load_eval_corpus(directory)
        queries = load_eval_queries(directory)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    documents = passages_as_documents(passages)
    secrets = _secrets_from_env()
    embedder = build_embedder(
        embedder_name,
        api_key=_env("OPENAI_API_KEY") if embedder_name == "openai" else "",
        model=_env("OPENAI_EMBEDDING_MODEL") if embedder_name == "openai" else "",
        base_url=_env("OPENAI_BASE_URL") if embedder_name == "openai" else "",
    )
    needs_rerank = any(split_pipeline(name)[1] for name in pipelines)
    reranker = build_reranker(reranker_name) if needs_rerank else None
    records: list[EvalRecord] = []
    for name in pipelines:
        _first_stage, use_rerank = split_pipeline(name)
        records.extend(
            evaluate_pipeline(
                name,
                documents,
                queries,
                retriever=make_retriever(_first_stage, embedder),
                reranker=reranker if use_rerank else None,
                rerank_top_n=args.rerank_top_n,
                embedder_name=embedder_name,
                reranker_name=reranker_name,
                secrets=secrets,
            )
        )
    eval_dir = DEFAULT_EVAL_DIR if directory is None else Path(directory).as_posix()
    summary = build_summary(
        records,
        eval_dir=eval_dir,
        embedder=embedder_name,
        reranker=reranker_name,
        rerank_top_n=args.rerank_top_n,
        passage_count=len(passages),
        query_count=len(queries),
    )
    write_evaluation(args.output, records, summary)
    return records, args.output


def _choice(value: str, options: tuple[str, ...], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in options:
        names = ", ".join(options)
        raise ConfigError(f"{label} inválido: {value}. Use: {names}.")
    return normalized


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _secrets_from_env() -> tuple[str, ...]:
    secret = _env("OPENAI_API_KEY")
    if len(secret) >= 8:
        return (secret,)
    return ()


def _success_record(
    identity: _Identity,
    query: EvalQuery,
    result: PipelineResult,
    counter: _CountingReranker | None,
) -> EvalRecord:
    final_ids = tuple(result.ids)
    first_ids = tuple(hit.doc_id for hit in result.first_stage_hits)
    scored = _score(final_ids, query.relevance)
    # Sem rerank o lift não existe. Zero só aparece quando o rerank não mudou o nDCG@10.
    if identity.rerank:
        lift = scored["ndcg_at_10"] - ndcg_at_k(first_ids, query.relevance, RETRIEVAL_K)
    else:
        lift = None
    latency = {
        "retrieve": float(result.stage_latency_ms["retrieve"]),
        "rerank": (
            float(result.stage_latency_ms["rerank"])
            if "rerank" in result.stage_latency_ms
            else None
        ),
    }
    prompt, completion, cost = _combine_usage(None if counter is None else counter.usage)
    return EvalRecord(
        embedder=identity.embedder,
        reranker=identity.reranker,
        pipeline=identity.pipeline,
        first_stage=identity.first_stage,
        rerank=identity.rerank,
        rerank_top_n=identity.rerank_top_n,
        query_id=query.query_id,
        category=query.category,
        retrieved_ids=final_ids,
        first_stage_ids=first_ids,
        recall_at_1=scored["recall_at_1"],
        recall_at_3=scored["recall_at_3"],
        recall_at_5=scored["recall_at_5"],
        recall_at_10=scored["recall_at_10"],
        mrr_at_10=scored["mrr_at_10"],
        ndcg_at_5=scored["ndcg_at_5"],
        ndcg_at_10=scored["ndcg_at_10"],
        ndcg_at_10_lift=lift,
        stage_latency_ms=latency,
        reranker_calls=0 if counter is None else counter.calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost,
        error=None,
    )


def _error_record(
    identity: _Identity,
    query: EvalQuery,
    message: str,
    *,
    calls: int,
    usage: ModelUsage | None,
) -> EvalRecord:
    prompt, completion, cost = _combine_usage(usage)
    return EvalRecord(
        embedder=identity.embedder,
        reranker=identity.reranker,
        pipeline=identity.pipeline,
        first_stage=identity.first_stage,
        rerank=identity.rerank,
        rerank_top_n=identity.rerank_top_n,
        query_id=query.query_id,
        category=query.category,
        retrieved_ids=None,
        first_stage_ids=None,
        recall_at_1=None,
        recall_at_3=None,
        recall_at_5=None,
        recall_at_10=None,
        mrr_at_10=None,
        ndcg_at_5=None,
        ndcg_at_10=None,
        ndcg_at_10_lift=None,
        stage_latency_ms=None,
        reranker_calls=calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost,
        error=message,
    )


def _score(ids: Sequence[str], relevance: Mapping[str, int]) -> dict[str, float]:
    scored = {f"recall_at_{k}": recall_at_k(ids, relevance, k) for k in RECALL_KS}
    scored["mrr_at_10"] = mrr(list(ids)[:MRR_K], relevance)
    for k in NDCG_KS:
        scored[f"ndcg_at_{k}"] = ndcg_at_k(ids, relevance, k)
    return scored


def _group_key(record: EvalRecord) -> tuple[object, ...]:
    return (record.embedder, record.reranker, record.pipeline, record.rerank_top_n)


def _group_sort(key: tuple[object, ...]) -> tuple[object, ...]:
    embedder, reranker, pipeline, top_n = key
    if pipeline in PIPELINES:
        order = PIPELINES.index(str(pipeline))
    else:
        order = len(PIPELINES)
    missing = top_n is None
    number = top_n if isinstance(top_n, int) else 0
    return (str(embedder), str(reranker), order, missing, number)


def _categories(records: Sequence[EvalRecord]) -> list[dict[str, object]]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.category, []).append(record)
    names = [name for name in CATEGORY_ORDER if name in grouped]
    names.extend(sorted(name for name in grouped if name not in names))
    blocks: list[dict[str, object]] = []
    for name in names:
        block = _metrics_block(grouped[name], category=name)
        blocks.append(block)
    return blocks


def _metrics_block(records: Sequence[EvalRecord], *, category: str | None) -> dict[str, object]:
    head = records[0]
    block: dict[str, object]
    if category is None:
        block = {
            "embedder": head.embedder,
            "reranker": head.reranker,
            "pipeline": head.pipeline,
            "first_stage": head.first_stage,
            "rerank": head.rerank,
            "rerank_top_n": head.rerank_top_n,
        }
    else:
        block = {"category": category}
    block["query_count"] = len(records)
    block["error_count"] = sum(1 for record in records if record.error is not None)
    for name in _METRIC_KEYS:
        block[name] = _mean_defined(_metric_values(records, name))
    block["stage_latency_ms"] = _latency_block(records)
    block["reranker_calls"] = sum(record.reranker_calls for record in records)
    block["prompt_tokens"] = _sum_tokens([record.prompt_tokens for record in records])
    block["completion_tokens"] = _sum_tokens([record.completion_tokens for record in records])
    block["cost_usd"] = _sum_costs([record.cost_usd for record in records])
    if category is not None and tuple(block) != CATEGORY_KEYS:
        missing = set(CATEGORY_KEYS) - set(block)
        raise RuntimeError(f"schema da categoria divergiu: {sorted(missing)}.")
    return block


def _metric_values(records: Sequence[EvalRecord], name: str) -> list[float | None]:
    # Erro sai mesmo se a linha trouxer número. Null de sucesso continua de fora, zero fica.
    values: list[float | None] = []
    for record in records:
        if record.error is not None:
            values.append(None)
            continue
        value = getattr(record, name)
        values.append(None if value is None else float(value))
    return values


def _mean_defined(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined) / len(defined)


def _latency_block(records: Sequence[EvalRecord]) -> dict[str, dict[str, float | int | None]]:
    return {stage: _stage_stats(records, stage) for stage in _LATENCY_STAGES}


def _stage_stats(records: Sequence[EvalRecord], stage: str) -> dict[str, float | int | None]:
    values: list[float] = []
    for record in records:
        if record.error is not None or record.stage_latency_ms is None:
            continue
        value = record.stage_latency_ms.get(stage)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return {"count": 0, "mean": None}
    return {"count": len(values), "mean": sum(values) / len(values)}


def _sum_tokens(values: Sequence[int | None]) -> int | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return sum(defined)


def _sum_costs(values: Sequence[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    if not defined:
        return None
    return float(sum(defined))


def _usage_of(component: object) -> ModelUsage | None:
    raw = getattr(component, "last_usage", None)
    if isinstance(raw, ModelUsage):
        return raw
    return None


def _combine_usage(usage: ModelUsage | None) -> tuple[int | None, int | None, float | None]:
    if usage is None:
        return None, None, None
    return usage.prompt_tokens, usage.completion_tokens, usage.cost_usd


def _public_error(exc: BaseException, secrets: Sequence[str]) -> str:
    text = _redact(f"{type(exc).__name__}: {exc}", secrets)
    if len(text) > 500:
        return text[:500] + "..."
    return text


def _redact(text: str, secrets: Sequence[str]) -> str:
    redacted = text
    for secret in secrets:
        if len(secret) >= 8:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _l2_normalize(vector: list[float]) -> tuple[float, ...]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def _dumps_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _dumps_jsonl(rows: Sequence[Mapping[str, object]]) -> str:
    lines = (
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows
    )
    return "".join(lines)


def _text(value: object, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _required_bool(value: object, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _required_int(value: object, key: str) -> int:
    parsed = _optional_int(value, key)
    if parsed is None:
        raise ConfigError(f"Campo {key} inválido no registro.")
    return parsed


def _optional_int(value: object, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return value


def _optional_float(value: object, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return float(value)


def _ids(value: object, key: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"Campo {key} inválido no registro.")
    if not all(isinstance(item, str) for item in value):
        raise ConfigError(f"Campo {key} inválido no registro.")
    return tuple(value)


def _latency(value: object) -> dict[str, float | None] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("Campo stage_latency_ms inválido no registro.")
    parsed: dict[str, float | None] = {}
    for key, item in value.items():
        if key not in _LATENCY_STAGES:
            raise ConfigError("Campo stage_latency_ms inválido no registro.")
        parsed[str(key)] = _optional_float(item, key)
    return parsed


if __name__ == "__main__":
    main()
