"""Configuração lida do ambiente. Segredos não têm default e não são logados.

Modelo vazio cai no default de código; a chave da API permanece vazia até o .env local.
"""

import os
from dataclasses import dataclass

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Settings:
    """Parâmetros dos rerankers opcionais. Campos de chave podem ser string vazia."""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    cross_encoder_model: str


def load_settings() -> Settings:
    """Lê o processo atual. Não abre arquivo .env, para não puxar segredo sem intenção."""

    return Settings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", ""),
        openai_model=os.environ.get("OPENAI_MODEL", "") or DEFAULT_OPENAI_MODEL,
        cross_encoder_model=os.environ.get("CROSS_ENCODER_MODEL", "")
        or DEFAULT_CROSS_ENCODER_MODEL,
    )
