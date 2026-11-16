"""Índice em memória dos documentos do primeiro estágio.

A última escrita de um mesmo id substitui a anterior, então reindexar troca o corpus.
"""

from collections.abc import Sequence

from reranking_study.types import Document


class InMemoryIndex:
    """Mapa id → documento. Não persiste em disco; o corpus de teste cabe no processo."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def upsert(self, documents: Sequence[Document]) -> None:
        for document in documents:
            self._documents[document.doc_id] = document

    def __len__(self) -> int:
        return len(self._documents)

    def documents(self) -> list[Document]:
        return list(self._documents.values())
