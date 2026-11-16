"""Rerankers do segundo estágio. O falso é determinístico; os reais só importam na chamada viva.

sentence-transformers e openai ficam nos extras. Import no topo quebraria o teste sem essas libs.
"""

import json
from collections.abc import Sequence
from typing import Protocol

from reranking_study.config import load_settings
from reranking_study.tokenize import tokenize
from reranking_study.types import Hit


class Reranker(Protocol):
    """Reordena um conjunto já recuperado. Não busca documento fora dessa lista."""

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]: ...


class FakeReranker:
    """Ordena pelo tamanho da interseção de tokens entre consulta e documento.

    Empate desempata pelo doc_id. Serve ao teste; não estima relevância semântica.
    """

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        _validate_top_n(top_n)
        if top_n == 0 or not hits:
            return []
        query_terms = set(tokenize(query))
        ranked = [
            Hit(hit.doc_id, float(len(query_terms & set(tokenize(hit.text)))), hit.text)
            for hit in hits
        ]
        ranked.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return ranked[:top_n]


class CrossEncoderReranker:
    """Cross-encoder via sentence-transformers. O modelo só carrega na primeira chamada de rerank.

    Sem o extra cross-encoder, a chamada levanta ImportError em vez de devolver escore inventado.
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = load_settings()
        self.model_name = model_name or settings.cross_encoder_model
        self._model: object | None = None

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        _validate_top_n(top_n)
        if top_n == 0 or not hits:
            return []
        model = self._get_model()
        pairs = [(query, hit.text) for hit in hits]
        raw_scores = model.predict(pairs)
        ranked = [
            Hit(hit.doc_id, float(score), hit.text)
            for hit, score in zip(hits, raw_scores, strict=True)
        ]
        ranked.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return ranked[:top_n]

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "CrossEncoderReranker requires the optional 'cross-encoder' extra"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        return self._model


class LlmReranker:
    """LLM-as-reranker numa única chamada. O cliente OpenAI nasce na primeira chamada.

    Cada id recebido precisa voltar com escore. Id omitido é erro, não escore zero implícito.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        settings = load_settings()
        self.model = model or settings.openai_model
        self.api_key = settings.openai_api_key if api_key is None else api_key
        self.base_url = settings.openai_base_url if base_url is None else base_url
        self._client: object | None = None

    def rerank(self, query: str, hits: Sequence[Hit], top_n: int) -> list[Hit]:
        _validate_top_n(top_n)
        if top_n == 0 or not hits:
            return []
        client = self._get_client()
        catalog = "\n".join(f"{hit.doc_id}\t{hit.text}" for hit in hits)
        prompt = (
            "Score how relevant each document is to the query. "
            'Reply with JSON only: {"scores": {"doc_id": number}}. '
            "Higher means more relevant. Include every document id.\n"
            f"Query: {query}\n"
            f"Documents:\n{catalog}"
        )
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        scores = _parse_score_map(content)
        missing = [hit.doc_id for hit in hits if hit.doc_id not in scores]
        if missing:
            raise ValueError(f"LLM reranker omitted document ids: {', '.join(missing)}")
        ranked = [Hit(hit.doc_id, scores[hit.doc_id], hit.text) for hit in hits]
        ranked.sort(key=lambda hit: (-hit.score, hit.doc_id))
        return ranked[:top_n]

    def _get_client(self) -> object:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is empty")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("LlmReranker requires the optional 'llm' extra") from exc
            kwargs: dict[str, str] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client


def _parse_score_map(content: str) -> dict[str, float]:
    """Extrai o objeto scores. Cerca de markdown é aceita; JSON inválido ou sem scores falha."""

    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("LLM reranker response did not contain a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict) or "scores" not in payload:
        raise ValueError("LLM reranker JSON must contain a 'scores' object")
    scores = payload["scores"]
    if not isinstance(scores, dict):
        raise ValueError("LLM reranker 'scores' must be an object")
    return {str(doc_id): float(value) for doc_id, value in scores.items()}


def _validate_top_n(top_n: int) -> None:
    if top_n < 0:
        raise ValueError("top_n must be >= 0")
