"""Contrato do harness: a média ignora erro e null, e conserva o zero.

O hash confere o pipeline. Componente real sem --allow-live ou sem variável
sai com código 2 e não chama rede. Nenhum número daqui é achado do estudo.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reranking_study.eval_data import load_eval_queries
from reranking_study.evaluate import (
    AGGREGATE_KEYS,
    CATEGORY_KEYS,
    CATEGORY_ORDER,
    FAKE_DISCLAIMER,
    PIPELINES,
    RECORD_KEYS,
    RECORDS_FILENAME,
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    SUMMARY_KEYS,
    EvalRecord,
    ModelUsage,
    OpenAiEmbedder,
    aggregate_records,
    build_embedder,
    build_reranker,
    eval_record_from_dict,
    evaluate_pipeline,
    main,
    parse_args,
    record_to_dict,
)
from reranking_study.metrics import mrr, ndcg_at_k, recall_at_k
from reranking_study.types import Document, Hit

ROOT = Path(__file__).resolve().parents[1]
SECRET = "super-secret-value"
OTHER_SECRET = "another-secret-value"


def _row(**overrides: object) -> EvalRecord:
    base: dict[str, object] = {
        "embedder": "fake",
        "reranker": "fake",
        "pipeline": "dense",
        "first_stage": "dense",
        "rerank": False,
        "rerank_top_n": None,
        "query_id": "q",
        "category": "lexical",
        "retrieved_ids": (),
        "first_stage_ids": (),
        "recall_at_1": None,
        "recall_at_3": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "mrr_at_10": None,
        "ndcg_at_5": None,
        "ndcg_at_10": None,
        "ndcg_at_10_lift": None,
        "stage_latency_ms": None,
        "reranker_calls": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_usd": None,
        "error": None,
    }
    base.update(overrides)
    return EvalRecord(**base)  # type: ignore[arg-type]


def _scored(query_id: str, category: str, recall: float | None, **overrides: object) -> EvalRecord:
    return _row(
        query_id=query_id,
        category=category,
        recall_at_1=recall,
        recall_at_3=recall,
        recall_at_5=recall,
        recall_at_10=recall,
        **overrides,
    )


def test_aggregate_drops_error_keeps_zero_and_averages_queries() -> None:
    # 1, 0 e 1, com o erro de fora, dá 2/3. A média das categorias daria 0,75.
    records = [
        _scored("a", "lexical", 1.0, mrr_at_10=1.0, ndcg_at_10=0.5),
        _scored("b", "lexical", 0.0, mrr_at_10=0.0, ndcg_at_10=0.0),
        _scored("c", "paraphrase", 1.0, mrr_at_10=0.5, ndcg_at_10=1.0),
        _scored(
            "d",
            "paraphrase",
            1.0,
            mrr_at_10=1.0,
            ndcg_at_10=1.0,
            error="falhou",
            retrieved_ids=None,
        ),
    ]
    [block] = aggregate_records(records)
    assert block["query_count"] == 4
    assert block["error_count"] == 1
    assert block["recall_at_1"] == pytest.approx(2 / 3)
    assert block["recall_at_10"] == pytest.approx(2 / 3)
    assert block["mrr_at_10"] == pytest.approx(0.5)
    assert block["ndcg_at_10"] == pytest.approx(0.5)
    assert set(AGGREGATE_KEYS) <= set(block)
    by_category = {item["category"]: item for item in block["by_category"]}
    assert list(by_category) == ["lexical", "paraphrase"]
    assert by_category["lexical"]["recall_at_1"] == pytest.approx(0.5)
    assert by_category["lexical"]["query_count"] == 2
    assert by_category["lexical"]["error_count"] == 0
    assert by_category["paraphrase"]["recall_at_1"] == pytest.approx(1.0)
    assert by_category["paraphrase"]["error_count"] == 1
    assert by_category["paraphrase"]["query_count"] == 2
    assert set(CATEGORY_KEYS) <= set(by_category["lexical"])


def test_aggregate_is_null_when_every_row_errored() -> None:
    records = [
        _scored("a", "distractor", 1.0, error="falhou", reranker_calls=1),
        _scored("b", "distractor", 0.0, error="falhou", reranker_calls=1),
    ]
    [block] = aggregate_records(records)
    assert block["recall_at_1"] is None
    assert block["recall_at_10"] is None
    assert block["mrr_at_10"] is None
    assert block["ndcg_at_5"] is None
    assert block["ndcg_at_10"] is None
    assert block["ndcg_at_10_lift"] is None
    assert block["prompt_tokens"] is None
    assert block["completion_tokens"] is None
    assert block["cost_usd"] is None
    assert block["error_count"] == 2
    assert block["query_count"] == 2
    assert block["reranker_calls"] == 2
    latency = block["stage_latency_ms"]
    assert latency["retrieve"] == {"count": 0, "mean": None}
    assert latency["rerank"] == {"count": 0, "mean": None}


def test_lift_stays_null_without_rerank_and_keeps_a_zero_delta() -> None:
    plain = [
        _row(query_id="a", ndcg_at_10=0.2, ndcg_at_10_lift=None),
        _row(query_id="b", ndcg_at_10=0.4, ndcg_at_10_lift=None),
    ]
    [block] = aggregate_records(plain)
    assert block["ndcg_at_10"] == pytest.approx(0.3)
    assert block["ndcg_at_10_lift"] is None

    reranked = [
        _row(
            query_id="a",
            pipeline="dense_rerank",
            first_stage="dense",
            rerank=True,
            rerank_top_n=10,
            ndcg_at_10=0.4,
            ndcg_at_10_lift=0.0,
        ),
        _row(
            query_id="b",
            pipeline="dense_rerank",
            first_stage="dense",
            rerank=True,
            rerank_top_n=10,
            ndcg_at_10=0.6,
            ndcg_at_10_lift=0.2,
        ),
    ]
    [block] = aggregate_records(reranked)
    assert block["ndcg_at_10_lift"] == pytest.approx(0.1)
    assert block["rerank"] is True
    assert block["rerank_top_n"] == 10


def test_latency_ignores_errors_and_reported_usage_still_counts() -> None:
    records = [
        _row(
            query_id="a",
            recall_at_10=1.0,
            stage_latency_ms={"retrieve": 10.0, "rerank": None},
            reranker_calls=0,
        ),
        _row(
            query_id="b",
            recall_at_10=0.0,
            stage_latency_ms={"retrieve": 30.0, "rerank": 6.0},
            reranker_calls=1,
        ),
        _row(
            query_id="c",
            recall_at_10=1.0,
            error="falhou",
            stage_latency_ms={"retrieve": 1000.0, "rerank": 1000.0},
            reranker_calls=1,
            prompt_tokens=4,
            completion_tokens=0,
            cost_usd=0.0,
        ),
    ]
    [block] = aggregate_records(records)
    assert block["recall_at_10"] == pytest.approx(0.5)
    assert block["stage_latency_ms"]["retrieve"]["count"] == 2
    assert block["stage_latency_ms"]["retrieve"]["mean"] == pytest.approx(20.0)
    assert block["stage_latency_ms"]["rerank"]["count"] == 1
    assert block["stage_latency_ms"]["rerank"]["mean"] == pytest.approx(6.0)
    assert block["reranker_calls"] == 2
    assert block["prompt_tokens"] == 4
    assert block["completion_tokens"] == 0
    assert block["cost_usd"] == pytest.approx(0.0)


def test_aggregate_orders_pipelines_and_does_not_mix_top_n() -> None:
    records = [
        _scored("late", "multi", 0.0, pipeline="hybrid_rrf_rerank", rerank=True, rerank_top_n=10),
        _scored("early", "lexical", 1.0, pipeline="bm25", first_stage="bm25"),
        _scored(
            "narrow",
            "lexical",
            1.0,
            pipeline="dense_rerank",
            rerank=True,
            rerank_top_n=3,
        ),
        _scored(
            "wide",
            "lexical",
            0.0,
            pipeline="dense_rerank",
            rerank=True,
            rerank_top_n=10,
        ),
    ]
    blocks = aggregate_records(records)
    assert [(block["pipeline"], block["rerank_top_n"]) for block in blocks] == [
        ("bm25", None),
        ("dense_rerank", 3),
        ("dense_rerank", 10),
        ("hybrid_rrf_rerank", 10),
    ]
    assert blocks[1]["recall_at_10"] == pytest.approx(1.0)
    assert blocks[2]["recall_at_10"] == pytest.approx(0.0)


def test_record_schema_round_trip() -> None:
    record = _scored(
        "q1",
        "multi",
        0.5,
        retrieved_ids=("d1", "d2"),
        first_stage_ids=("d2", "d1"),
        stage_latency_ms={"retrieve": 1.5, "rerank": None},
        mrr_at_10=1.0,
        ndcg_at_5=0.25,
        ndcg_at_10=0.5,
    )
    raw = record_to_dict(record)
    assert tuple(raw) == RECORD_KEYS
    assert raw["schema_version"] == SCHEMA_VERSION
    restored = eval_record_from_dict(raw)
    assert restored == record
    broken = dict(raw)
    del broken["query_id"]
    with pytest.raises(Exception, match="query_id"):
        eval_record_from_dict(broken)


class _FixedRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits

    def index(self, documents: list[Document]) -> None:
        return None

    def retrieve(self, query: str, k: int) -> list[Hit]:
        return list(self.hits[:k])


def test_rerank_lift_uses_graded_gain_against_the_same_first_stage() -> None:
    # Grau 2 e grau 1. Se os dois virassem relevante binário, a troca de ordem não mudaria o nDCG.
    hits = [Hit("d1", 0.2, "um"), Hit("d2", 0.1, "dois")]
    relevance = {"d2": 2, "d1": 1}
    query = _query("q", "pergunta", "lexical", relevance)

    class _Reverse:
        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            return list(reversed(list(hits[:top_n])))

    plain = evaluate_pipeline(
        "dense",
        [Document("d1", "um"), Document("d2", "dois")],
        [query],
        retriever=_FixedRetriever(hits),
        reranker=None,
        rerank_top_n=10,
        embedder_name="fake",
        reranker_name="fake",
    )[0]
    reranked = evaluate_pipeline(
        "dense_rerank",
        [Document("d1", "um"), Document("d2", "dois")],
        [query],
        retriever=_FixedRetriever(hits),
        reranker=_Reverse(),
        rerank_top_n=10,
        embedder_name="fake",
        reranker_name="fake",
    )[0]
    assert plain.ndcg_at_10_lift is None
    assert plain.retrieved_ids == ("d1", "d2")
    assert plain.first_stage_ids == ("d1", "d2")
    assert plain.reranker_calls == 0
    assert plain.prompt_tokens is None
    assert reranked.retrieved_ids == ("d2", "d1")
    assert reranked.first_stage_ids == ("d1", "d2")
    assert reranked.ndcg_at_10_lift == pytest.approx(
        ndcg_at_k(("d2", "d1"), relevance, 10) - ndcg_at_k(("d1", "d2"), relevance, 10)
    )
    assert reranked.ndcg_at_10_lift is not None
    assert reranked.ndcg_at_10_lift > 0
    assert reranked.reranker_calls == 1
    assert reranked.cost_usd is None


def test_rerank_top_n_limits_the_pool_and_copies_reported_usage() -> None:
    hits = [Hit(f"d{i}", float(i), "t") for i in range(5)]

    class _Spy:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            self.sizes.append((len(hits), top_n))
            self.last_usage = ModelUsage(prompt_tokens=5, completion_tokens=None, cost_usd=None)
            return list(hits)

    spy = _Spy()
    [record] = evaluate_pipeline(
        "bm25_rerank",
        [Document("d0", "t")],
        [_query("q", "t", "distractor", {"d0": 2})],
        retriever=_FixedRetriever(hits),
        reranker=spy,
        rerank_top_n=3,
        embedder_name="fake",
        reranker_name="fake",
    )
    assert spy.sizes == [(3, 3)]
    assert record.rerank_top_n == 3
    assert record.first_stage == "bm25"
    assert record.prompt_tokens == 5
    assert record.completion_tokens is None
    assert record.reranker_calls == 1


def test_query_error_is_redacted_and_excluded_from_the_mean() -> None:
    class _Selective:
        def index(self, documents: list[Document]) -> None:
            return None

        def retrieve(self, query: str, k: int) -> list[Hit]:
            if "BOOM" in query:
                raise ConnectionError(f"reset {SECRET}")
            return [Hit("d1", 1.0, "alfa")]

    records = evaluate_pipeline(
        "dense",
        [Document("d1", "alfa")],
        [
            _query("good", "alfa", "lexical", {"d1": 2}),
            _query("bad", "BOOM rede", "paraphrase", {"d1": 2}),
        ],
        retriever=_Selective(),
        reranker=None,
        rerank_top_n=10,
        embedder_name="fake",
        reranker_name="fake",
        secrets=(SECRET,),
    )
    assert [record.query_id for record in records] == ["good", "bad"]
    assert records[0].error is None
    assert records[0].recall_at_10 == pytest.approx(1.0)
    assert records[1].recall_at_10 is None
    assert records[1].retrieved_ids is None
    assert records[1].error is not None
    assert SECRET not in records[1].error
    assert "[redacted]" in records[1].error
    [block] = aggregate_records(records)
    assert block["recall_at_10"] == pytest.approx(1.0)
    assert block["error_count"] == 1
    assert block["query_count"] == 2
    paraphrase = block["by_category"][1]
    assert paraphrase["category"] == "paraphrase"
    assert paraphrase["recall_at_10"] is None


def test_failed_rerank_still_counts_as_a_call() -> None:
    class _Boom:
        def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
            raise TimeoutError(f"slow {SECRET}")

    [record] = evaluate_pipeline(
        "dense_rerank",
        [Document("d1", "alfa")],
        [_query("q", "alfa", "lexical", {"d1": 2})],
        retriever=_FixedRetriever([Hit("d1", 1.0, "alfa")]),
        reranker=_Boom(),
        rerank_top_n=10,
        embedder_name="fake",
        reranker_name="fake",
        secrets=(SECRET,),
    )
    assert record.reranker_calls == 1
    assert record.recall_at_10 is None
    assert record.stage_latency_ms is None
    assert record.error is not None
    assert SECRET not in record.error
    assert "[redacted]" in record.error


def test_index_error_nulls_every_query_and_does_not_retrieve() -> None:
    class _BadIndex:
        def index(self, documents: list[Document]) -> None:
            raise RuntimeError(f"index {SECRET}")

        def retrieve(self, query: str, k: int) -> list[Hit]:
            raise AssertionError("retrieve")

    records = evaluate_pipeline(
        "hybrid_rrf",
        [Document("d1", "alfa")],
        [_query("q1", "alfa", "multi", {"d1": 2}), _query("q2", "beta", "multi", {"d1": 2})],
        retriever=_BadIndex(),
        reranker=None,
        rerank_top_n=10,
        embedder_name="fake",
        reranker_name="fake",
        secrets=(SECRET,),
    )
    assert len(records) == 2
    assert all(record.error is not None and SECRET not in record.error for record in records)
    assert all(record.recall_at_1 is None and record.reranker_calls == 0 for record in records)
    [block] = aggregate_records(records)
    assert block["recall_at_10"] is None
    assert block["error_count"] == 2


def test_openai_embedder_does_not_import_until_embed_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    embedder = OpenAiEmbedder(api_key="test-key", model="text-embedding-3-small", base_url="")
    with pytest.raises(ImportError, match="llm"):
        embedder.embed("texto")
    empty = OpenAiEmbedder(api_key="", model="m", base_url="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        empty.embed("texto")

    class _Data:
        embedding = [3.0, 4.0]

    class _Response:
        data = [_Data()]

    class _Embeddings:
        def create(self, **kwargs: object) -> _Response:
            assert kwargs["model"] == "text-embedding-3-small"
            assert kwargs["input"] == "texto"
            return _Response()

    class _Client:
        embeddings = _Embeddings()

    embedder._get_client = lambda: _Client()  # type: ignore[method-assign]
    assert embedder.embed("texto") == pytest.approx((0.6, 0.8))


def test_parser_defaults_to_fake_components() -> None:
    args = parse_args([])
    assert args.pipeline == "all"
    assert args.rerank_top_n == 10
    assert args.embedder == "fake"
    assert args.reranker == "fake"
    assert args.allow_live is False
    assert args.output == Path("results/eval")
    assert args.eval_dir is None


def _query(query_id: str, text: str, category: str, relevance: dict[str, int]) -> object:
    from reranking_study.eval_data import EvalQuery

    return EvalQuery(query_id, text, category, relevance)


def _strip_latency(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _strip_latency(item) for key, item in value.items() if key != "stage_latency_ms"
        }
    if isinstance(value, list):
        return [_strip_latency(item) for item in value]
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cli_fake_v1_is_deterministic_and_matches_the_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    first = tmp_path / "first"
    second = tmp_path / "second"
    main(["--pipeline", "all", "--output", str(first)])
    captured = capsys.readouterr()
    assert SECRET not in captured.out
    assert SECRET not in captured.err
    main(["--pipeline", "all", "--output", str(second)])

    first_rows = _load_jsonl(first / RECORDS_FILENAME)
    second_rows = _load_jsonl(second / RECORDS_FILENAME)
    first_summary = json.loads((first / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    second_summary = json.loads((second / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert _strip_latency(first_rows) == _strip_latency(second_rows)
    assert _strip_latency(first_summary) == _strip_latency(second_summary)

    queries = load_eval_queries()
    relevance = {query.query_id: query.relevance for query in queries}
    assert first_summary["schema_version"] == SCHEMA_VERSION
    assert set(first_summary) == set(SUMMARY_KEYS)
    assert first_summary["dataset"] == "v1"
    assert first_summary["eval_dir"] == "data/eval/v1"
    assert first_summary["embedder"] == "fake"
    assert first_summary["reranker"] == "fake"
    assert first_summary["fake_components"] is True
    assert first_summary["disclaimer"] == FAKE_DISCLAIMER
    assert first_summary["passage_count"] == 72
    assert first_summary["query_count"] == len(queries) == 48
    assert first_summary["row_count"] == len(queries) * len(PIPELINES)
    assert first_summary["retrieval_k"] == 10
    assert first_summary["recall_ks"] == [1, 3, 5, 10]
    assert [block["pipeline"] for block in first_summary["pipelines"]] == list(PIPELINES)
    assert [row["query_id"] for row in first_rows] == [
        query.query_id for _name in PIPELINES for query in queries
    ]
    assert [row["pipeline"] for row in first_rows] == [
        name for name in PIPELINES for _query in queries
    ]
    for block in first_summary["pipelines"]:
        assert set(block) == set(AGGREGATE_KEYS)
        assert block["query_count"] == 48
        assert block["error_count"] == 0
        assert block["prompt_tokens"] is None
        assert block["completion_tokens"] is None
        assert block["cost_usd"] is None
        assert [item["category"] for item in block["by_category"]] == list(CATEGORY_ORDER)
        assert sum(item["query_count"] for item in block["by_category"]) == 48
        retrieve = block["stage_latency_ms"]["retrieve"]
        rerank = block["stage_latency_ms"]["rerank"]
        assert retrieve["count"] == 48
        assert retrieve["mean"] >= 0
        if block["rerank"]:
            assert block["ndcg_at_10_lift"] is not None
            assert block["reranker_calls"] == 48
            assert block["rerank_top_n"] == 10
            assert rerank["count"] == 48
            assert rerank["mean"] >= 0
        else:
            assert block["ndcg_at_10_lift"] is None
            assert block["reranker_calls"] == 0
            assert block["rerank_top_n"] is None
            assert rerank == {"count": 0, "mean": None}
    for row in first_rows:
        assert set(row) == set(RECORD_KEYS)
        assert row["error"] is None
        assert row["schema_version"] == SCHEMA_VERSION
        assert SECRET not in json.dumps(row)
        assert len(row["retrieved_ids"]) == 10
        grades = relevance[row["query_id"]]
        for k in (1, 3, 5, 10):
            assert row[f"recall_at_{k}"] == pytest.approx(
                recall_at_k(row["retrieved_ids"], grades, k)
            )
        assert row["recall_at_1"] <= row["recall_at_3"] <= row["recall_at_5"] <= row["recall_at_10"]
        assert row["mrr_at_10"] == pytest.approx(mrr(row["retrieved_ids"][:10], grades))
        assert row["ndcg_at_5"] == pytest.approx(ndcg_at_k(row["retrieved_ids"], grades, 5))
        assert row["ndcg_at_10"] == pytest.approx(ndcg_at_k(row["retrieved_ids"], grades, 10))
        if row["rerank"]:
            assert row["ndcg_at_10_lift"] == pytest.approx(
                ndcg_at_k(row["retrieved_ids"], grades, 10)
                - ndcg_at_k(row["first_stage_ids"], grades, 10)
            )
        else:
            assert row["ndcg_at_10_lift"] is None
            assert row["first_stage_ids"] == row["retrieved_ids"]
        assert row["prompt_tokens"] is None
        assert row["cost_usd"] is None
    restored = [eval_record_from_dict(row) for row in first_rows]
    assert aggregate_records(restored) == first_summary["pipelines"]
    blob = (first / SUMMARY_FILENAME).read_text(encoding="utf-8")
    assert SECRET not in blob
    assert "Registros:" in captured.out


def test_cli_rerank_top_n_is_stored(tmp_path: Path) -> None:
    output = tmp_path / "n4"
    main(["--pipeline", "dense_rerank", "--rerank-top-n", "4", "--output", str(output)])
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    rows = _load_jsonl(output / RECORDS_FILENAME)
    assert summary["rerank_top_n"] == 4
    assert summary["pipelines"][0]["rerank_top_n"] == 4
    assert rows[0]["rerank_top_n"] == 4
    assert len(rows) == 48


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("rede")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


def _forbid_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("componente live construído")

    monkeypatch.setattr("reranking_study.evaluate.build_embedder", _boom)
    monkeypatch.setattr("reranking_study.evaluate.build_reranker", _boom)


def _clear_live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "CROSS_ENCODER_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (["--embedder", "openai"], {"OPENAI_API_KEY": SECRET, "OPENAI_EMBEDDING_MODEL": "m"}),
        (["--reranker", "llm"], {"OPENAI_API_KEY": SECRET, "OPENAI_MODEL": "gpt-4o-mini"}),
        (["--reranker", "cross-encoder"], {"CROSS_ENCODER_MODEL": "cross-encoder/fake"}),
    ],
)
def test_live_components_without_flag_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    env: dict[str, str],
) -> None:
    _clear_live_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _forbid_network(monkeypatch)
    _forbid_builders(monkeypatch)
    output = tmp_path / "nope"
    with pytest.raises(SystemExit) as caught:
        main([*argv, "--output", str(output)])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert "--allow-live" in err
    assert "Nenhuma chamada foi feita." in err
    assert "Faltam variáveis" not in err
    assert SECRET not in err
    assert not output.exists()


@pytest.mark.parametrize(
    ("argv", "env", "missing"),
    [
        (
            ["--embedder", "openai", "--allow-live"],
            {"OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"},
            "OPENAI_API_KEY",
        ),
        (
            ["--embedder", "openai", "--allow-live"],
            {"OPENAI_API_KEY": SECRET},
            "OPENAI_EMBEDDING_MODEL",
        ),
        (
            ["--reranker", "llm", "--allow-live"],
            {"OPENAI_API_KEY": SECRET},
            "OPENAI_MODEL",
        ),
        (
            ["--reranker", "cross-encoder", "--allow-live"],
            {},
            "CROSS_ENCODER_MODEL",
        ),
    ],
)
def test_live_components_without_env_exit_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    env: dict[str, str],
    missing: str,
) -> None:
    _clear_live_env(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _forbid_network(monkeypatch)
    _forbid_builders(monkeypatch)
    output = tmp_path / "nope"
    with pytest.raises(SystemExit) as caught:
        main([*argv, "--output", str(output)])
    assert caught.value.code == 2
    err = capsys.readouterr().err
    assert missing in err
    assert "Nenhuma chamada foi feita." in err
    assert SECRET not in err
    assert not output.exists()


def test_allow_live_with_credentials_uses_the_builders_and_hides_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_live_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.chdir(ROOT)
    _forbid_network(monkeypatch)
    seen: dict[str, object] = {}

    def _embedder(name: str, *, api_key: str, model: str, base_url: str) -> object:
        seen["embedder"] = name
        seen["api_key"] = api_key
        seen["model"] = model
        return build_embedder("fake", api_key="", model="", base_url="")

    def _reranker(name: str) -> object:
        seen["reranker"] = name
        return build_reranker("fake")

    monkeypatch.setattr("reranking_study.evaluate.build_embedder", _embedder)
    monkeypatch.setattr("reranking_study.evaluate.build_reranker", _reranker)
    before_openai = "openai" in sys.modules
    before_cross = "sentence_transformers" in sys.modules
    output = tmp_path / "live"
    main(
        [
            "--pipeline",
            "dense_rerank",
            "--embedder",
            "openai",
            "--reranker",
            "llm",
            "--allow-live",
            "--output",
            str(output),
        ]
    )
    assert seen["embedder"] == "openai"
    assert seen["api_key"] == SECRET
    assert seen["model"] == "text-embedding-3-small"
    assert seen["reranker"] == "llm"
    assert ("openai" in sys.modules) is before_openai
    assert ("sentence_transformers" in sys.modules) is before_cross
    summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    blob = (output / SUMMARY_FILENAME).read_text(encoding="utf-8") + (
        output / RECORDS_FILENAME
    ).read_text(encoding="utf-8")
    assert summary["embedder"] == "openai"
    assert summary["reranker"] == "llm"
    assert summary["fake_components"] is False
    assert SECRET not in blob
    assert OTHER_SECRET not in blob


def test_invalid_pipeline_and_top_n_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--pipeline", "rrf", "--output", str(tmp_path / "out")])
    assert caught.value.code == 1
    assert "pipeline inválido" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()

    with pytest.raises(SystemExit) as caught:
        main(["--pipeline", "dense", "--rerank-top-n", "0", "--output", str(tmp_path / "out2")])
    assert caught.value.code == 1
    assert "rerank-top-n" in capsys.readouterr().err
    assert not (tmp_path / "out2").exists()


def test_module_and_console_script_refuse_a_live_provider(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = SECRET
    env["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
    module = subprocess.run(
        [
            sys.executable,
            "-m",
            "reranking_study.evaluate",
            "--embedder",
            "openai",
            "--output",
            str(tmp_path / "mod"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert module.returncode == 2, module.stderr
    assert "--allow-live" in module.stderr
    assert SECRET not in module.stderr
    assert not (tmp_path / "mod").exists()

    script = Path(sys.executable).with_name("evaluate-retrieval")
    assert script.is_file(), f"console script ausente em {script}"
    env["OPENAI_MODEL"] = "gpt-4o-mini"
    env["OPENAI_API_KEY"] = OTHER_SECRET
    console = subprocess.run(
        [str(script), "--reranker", "llm", "--output", str(tmp_path / "cli")],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert console.returncode == 2, console.stderr
    assert "--allow-live" in console.stderr
    assert OTHER_SECRET not in console.stderr
    assert not (tmp_path / "cli").exists()


def test_readme_and_gitignore_document_the_harness() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "evaluate-retrieval" in readme
    assert "python -m reranking_study.evaluate" in readme
    assert "--allow-live" in readme
    assert "código 2" in readme
    assert "Não são achados do estudo." in readme
    assert "Este README não publica recall, MRR, nDCG nem lift." in readme
    for key in (
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
    ):
        assert key in readme
    assert "results/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
