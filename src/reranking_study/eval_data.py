"""Conjunto de avaliação versionado: trechos, perguntas e gabarito.

O grau 0 não é gravado; ausência no mapa vale irrelevante. O loader recusa arquivo frouxo
para o experimento não herdar id quebrado, grau fora da escala ou categoria inventada.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

EVAL_CATEGORIES = frozenset({"lexical", "paraphrase", "distractor", "multi"})
ALLOWED_GRADES = frozenset({1, 2})
PASSAGES_FILENAME = "passages.jsonl"
QUERIES_FILENAME = "queries.jsonl"

_PASSAGE_FIELDS = frozenset({"id", "domain", "text", "near_duplicate_of"})
_PASSAGE_REQUIRED = frozenset({"id", "domain", "text"})
_QUERY_FIELDS = frozenset({"id", "category", "text", "relevance"})


def default_eval_dir() -> Path:
    """Diretório data/eval/v1 ao lado de src. O teste não depende do diretório de trabalho."""

    return Path(__file__).resolve().parents[2] / "data" / "eval" / "v1"


@dataclass(frozen=True)
class EvalPassage:
    """Trecho indexável do conjunto versionado.

    near_duplicate_of aponta o original quando o texto muda um fato e mantém o resto.
    """

    doc_id: str
    text: str
    domain: str
    near_duplicate_of: str | None = None


@dataclass(frozen=True)
class EvalQuery:
    """Pergunta com categoria fixa e graus 1 ou 2. O mapa é só leitura; grau ausente é 0."""

    query_id: str
    text: str
    category: str
    relevance: Mapping[str, int]


def load_eval_corpus(directory: Path | None = None) -> tuple[EvalPassage, ...]:
    """Lê passages.jsonl e recusa id repetido, campo fora do esquema ou quase-duplicata solta.

    O diretório default é data/eval/v1. Id de quase-duplicata tem de existir e ser o original.
    """

    root = _directory(directory)
    rows = _read_jsonl(root / PASSAGES_FILENAME)
    if not rows:
        raise ValueError("eval corpus is empty")
    passages: list[EvalPassage] = []
    seen: set[str] = set()
    for line_no, row in rows:
        passages.append(_passage_from_row(row, line_no, seen))
    by_id = {passage.doc_id: passage for passage in passages}
    for passage in passages:
        target = passage.near_duplicate_of
        if target is None:
            continue
        # Variante aponta só o original. Corrente esconde qual fato é o canônico.
        if target not in by_id:
            raise ValueError(f"near_duplicate_of points at unknown passage id: {target}")
        if target == passage.doc_id:
            raise ValueError(f"near_duplicate_of cannot point at itself: {passage.doc_id}")
        if by_id[target].near_duplicate_of is not None:
            raise ValueError(f"near_duplicate_of cannot chain: {passage.doc_id}")
    return tuple(passages)


def load_eval_queries(directory: Path | None = None) -> tuple[EvalQuery, ...]:
    """Lê queries.jsonl e exige categoria conhecida, grau permitido e ao menos um grau 2.

    Os ids de trecho são conferidos contra o corpus do mesmo diretório.
    """

    root = _directory(directory)
    known = {passage.doc_id for passage in load_eval_corpus(root)}
    rows = _read_jsonl(root / QUERIES_FILENAME)
    if not rows:
        raise ValueError("eval queries are empty")
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for line_no, row in rows:
        queries.append(_query_from_row(row, line_no, seen, known))
    return tuple(queries)


def _directory(directory: Path | None) -> Path:
    root = default_eval_dir() if directory is None else Path(directory)
    if not root.is_dir():
        raise ValueError(f"eval directory not found: {root}")
    return root


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, object]]]:
    if not path.is_file():
        raise ValueError(f"missing eval file: {path.name}")
    text = path.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        raise ValueError(f"{path.name} must not start with a BOM")
    rows: list[tuple[int, dict[str, object]]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_no}: empty line")
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no}: invalid JSON") from exc
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name}:{line_no}: expected a JSON object")
        rows.append((line_no, raw))
    return rows


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f"duplicate key: {key}")
        row[key] = value
    return row


def _passage_from_row(row: dict[str, object], line_no: int, seen: set[str]) -> EvalPassage:
    _reject_unknown(row, _PASSAGE_FIELDS, line_no, "passage")
    _require_fields(row, _PASSAGE_REQUIRED, line_no, "passage")
    doc_id = _slug(row["id"], line_no, "passage id")
    if doc_id in seen:
        raise ValueError(f"duplicate passage id: {doc_id}")
    seen.add(doc_id)
    domain = _slug(row["domain"], line_no, "domain")
    text = _nonempty_text(row["text"], line_no, "passage text")
    if "near_duplicate_of" in row:
        near_id: str | None = _slug(row["near_duplicate_of"], line_no, "near_duplicate_of")
    else:
        near_id = None
    return EvalPassage(doc_id, text, domain, near_id)


def _query_from_row(
    row: dict[str, object],
    line_no: int,
    seen: set[str],
    known: set[str],
) -> EvalQuery:
    _reject_unknown(row, _QUERY_FIELDS, line_no, "query")
    _require_fields(row, _QUERY_FIELDS, line_no, "query")
    query_id = _slug(row["id"], line_no, "query id")
    if query_id in seen:
        raise ValueError(f"duplicate query id: {query_id}")
    seen.add(query_id)
    category = row["category"]
    if not isinstance(category, str) or category not in EVAL_CATEGORIES:
        allowed = ", ".join(sorted(EVAL_CATEGORIES))
        raise ValueError(f"category must be one of {allowed}: {category!r}")
    text = _nonempty_text(row["text"], line_no, "query text")
    relevance = _relevance(row["relevance"], line_no, known)
    return EvalQuery(query_id, text, category, relevance)


def _relevance(raw: object, line_no: int, known: set[str]) -> Mapping[str, int]:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line_no}: relevance must be an object")
    if not raw:
        raise ValueError(f"line {line_no}: relevance is empty")
    grades: dict[str, int] = {}
    for doc_id, grade in raw.items():
        if not isinstance(doc_id, str) or not _is_slug(doc_id):
            raise ValueError(f"line {line_no}: invalid passage id in relevance: {doc_id!r}")
        if doc_id not in known:
            raise ValueError(f"unknown passage id in relevance: {doc_id}")
        # Escala gravada é 1 ou 2. Booleano passa como int no Python e por isso é recusado à parte.
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in ALLOWED_GRADES:
            allowed = ", ".join(str(item) for item in sorted(ALLOWED_GRADES))
            raise ValueError(f"grade must be one of {allowed}: {grade!r}")
        grades[doc_id] = grade
    if 2 not in grades.values():
        raise ValueError(f"line {line_no}: every query needs a grade 2 passage")
    return MappingProxyType(grades)


def _reject_unknown(
    row: Mapping[str, object], allowed: frozenset[str], line_no: int, kind: str
) -> None:
    extra = set(row) - allowed
    if extra:
        name = sorted(extra)[0]
        raise ValueError(f"line {line_no}: unknown {kind} field: {name}")


def _require_fields(
    row: Mapping[str, object], required: frozenset[str], line_no: int, kind: str
) -> None:
    missing = required - set(row)
    if missing:
        name = sorted(missing)[0]
        raise ValueError(f"line {line_no}: missing {kind} field: {name}")


def _slug(value: object, line_no: int, label: str) -> str:
    if not isinstance(value, str) or not _is_slug(value):
        raise ValueError(f"line {line_no}: {label} must be a slug: {value!r}")
    return value


def _is_slug(value: str) -> bool:
    return _ID_RE.fullmatch(value) is not None


def _nonempty_text(value: object, line_no: int, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_no}: {label} must be non-empty text")
    return value
