"""Tokenização única para BM25, embedder hash e o reranker determinístico.

O mesmo corte em casefold impede que cada estágio conte uma palavra de um jeito.
"""

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]
