"""Contrato dos rerankers: o falso ordena por overlap; os reais não importam no import do pacote.

Escore de LLM só entra se o JSON trouxer cada id. Chave vazia não dispara cliente.
"""

import builtins
import sys

import pytest

from reranking_study.config import load_settings
from reranking_study.rerankers import (
    CrossEncoderReranker,
    FakeReranker,
    LlmReranker,
    _parse_score_map,
)
from reranking_study.types import Hit


def test_fake_reranker_orders_by_token_overlap_then_id() -> None:
    hits = [
        Hit("c", 5.0, "nada"),
        Hit("a", 4.0, "alpha"),
        Hit("b", 1.0, "alpha beta gamma"),
    ]
    ranked = FakeReranker().rerank("alpha beta", hits, top_n=3)
    assert [hit.doc_id for hit in ranked] == ["b", "a", "c"]
    assert [hit.score for hit in ranked] == [2.0, 1.0, 0.0]


def test_fake_reranker_breaks_ties_by_id_and_honors_top_n() -> None:
    hits = [
        Hit("b", 0.2, "alpha zzz"),
        Hit("a", 0.9, "alpha"),
    ]
    ranked = FakeReranker().rerank("alpha", hits, top_n=1)
    assert [hit.doc_id for hit in ranked] == ["a"]
    assert FakeReranker().rerank("alpha", hits, top_n=0) == []
    with pytest.raises(ValueError):
        FakeReranker().rerank("alpha", hits, top_n=-1)


def test_cross_encoder_orders_by_model_scores_without_importing_on_init() -> None:
    sys.modules.pop("sentence_transformers", None)
    reranker = CrossEncoderReranker(model_name="fake-model")
    assert "sentence_transformers" not in sys.modules

    class _Model:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(len(document)) for _query, document in pairs]

    reranker._get_model = lambda: _Model()  # type: ignore[method-assign]
    ranked = reranker.rerank("q", [Hit("a", 0.0, "z"), Hit("b", 0.0, "zzzz")], top_n=2)
    assert [hit.doc_id for hit in ranked] == ["b", "a"]
    assert ranked[0].score == pytest.approx(4.0)


def test_cross_encoder_live_call_imports_the_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    with pytest.raises(ImportError, match="cross-encoder"):
        CrossEncoderReranker().rerank("q", [Hit("a", 0.0, "texto")], top_n=1)


def test_llm_reranker_orders_by_parsed_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = LlmReranker(api_key="test-key", model="fake")

    class _Message:
        content = '{"scores": {"b": 0.2, "a": 0.9}}'

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **_kwargs: object) -> _Response:
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(reranker, "_get_client", lambda: _Client())
    ranked = reranker.rerank("q", [Hit("b", 1.0, "bb"), Hit("a", 0.1, "aa")], top_n=2)
    assert [hit.doc_id for hit in ranked] == ["a", "b"]
    assert ranked[0].score == pytest.approx(0.9)


def test_llm_reranker_rejects_omitted_ids_and_invalid_json() -> None:
    assert _parse_score_map('prefix ```json\n{"scores": {"a": 1}}\n```') == {"a": 1.0}
    with pytest.raises(ValueError):
        _parse_score_map("sem json")
    with pytest.raises(ValueError):
        _parse_score_map('{"other": 1}')

    reranker = LlmReranker(api_key="test-key")

    class _Message:
        content = '{"scores": {"a": 1}}'

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def create(self, **_kwargs: object) -> _Response:
            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    reranker._get_client = lambda: _Client()  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="omitted"):
        reranker.rerank("q", [Hit("a", 0.0, "aa"), Hit("b", 0.0, "bb")], top_n=2)


def test_llm_reranker_live_call_imports_the_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            raise ImportError("simulated missing extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    with pytest.raises(ImportError, match="llm"):
        LlmReranker(api_key="test-key").rerank("q", [Hit("a", 0.0, "texto")], top_n=1)


def test_llm_reranker_refuses_empty_key_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def _guard(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai" or name.startswith("openai."):
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard)
    reranker = LlmReranker(api_key="")
    assert reranker.rerank("q", [Hit("a", 0.0, "aa")], top_n=0) == []
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        reranker.rerank("q", [Hit("a", 0.0, "aa")], top_n=1)
    assert imported == []


def test_settings_keep_the_api_key_empty_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("CROSS_ENCODER_MODEL", raising=False)
    settings = load_settings()
    assert settings.openai_api_key == ""
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
