"""Contrato do loader e invariantes do conjunto data/eval/v1.

O arquivo versionado precisa falhar fechado: id repetido, grau fora da escala ou rótulo
solto não pode passar. As categorias são propriedade do texto, não um escore de retriever.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

from reranking_study.eval_data import (
    ALLOWED_GRADES,
    EVAL_CATEGORIES,
    load_eval_corpus,
    load_eval_queries,
)
from reranking_study.tokenize import tokenize

_STOP = frozenset(
    {
        "a",
        "o",
        "os",
        "as",
        "um",
        "uma",
        "uns",
        "umas",
        "de",
        "do",
        "da",
        "dos",
        "das",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "num",
        "numa",
        "dum",
        "duma",
        "por",
        "para",
        "com",
        "sem",
        "sob",
        "sobre",
        "e",
        "ou",
        "que",
        "se",
        "ao",
        "aos",
        "à",
        "às",
        "é",
        "são",
        "ser",
        "foi",
        "era",
        "eram",
        "como",
        "mais",
        "não",
        "sim",
        "já",
        "também",
        "até",
        "entre",
        "pelo",
        "pela",
        "pelos",
        "pelas",
        "seu",
        "sua",
        "seus",
        "suas",
        "esse",
        "essa",
        "este",
        "esta",
        "isso",
        "isto",
        "aquele",
        "aquela",
        "quando",
        "onde",
        "qual",
        "quais",
        "quem",
        "quanto",
        "quanta",
        "quantos",
        "quantas",
        "há",
        "ter",
        "tem",
        "tinha",
        "está",
        "estão",
        "estar",
        "ficar",
        "fica",
        "ficam",
        "ele",
        "ela",
        "eles",
        "elas",
        "lhe",
        "lhes",
        "mas",
        "porque",
        "pois",
        "então",
        "muito",
        "muita",
        "muitos",
        "muitas",
        "cada",
        "todo",
        "toda",
        "todos",
        "todas",
        "outro",
        "outra",
        "outros",
        "outras",
        "após",
        "antes",
        "durante",
        "desde",
        "apenas",
        "somente",
        "só",
        "pode",
        "podem",
        "deve",
        "devem",
        "nem",
        "depois",
        "dessa",
        "desse",
        "deste",
        "desta",
        "daquele",
        "daquela",
        "nele",
        "nela",
    }
)

_DOMAINS = {
    "biblioteca-vila-serpa": 18,
    "relojoaria-lume": 18,
    "cooperativa-lanoso": 18,
    "museu-bondinho-vila-serpa": 18,
}


def _content(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOP}


def _coverage(query: str, passage: str) -> float:
    query_tokens = _content(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _content(passage)) / len(query_tokens)


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def test_v1_counts_grades_and_fictional_domains() -> None:
    passages = load_eval_corpus()
    queries = load_eval_queries()
    assert len(passages) == 72
    assert len(queries) == 48
    assert Counter(query.category for query in queries) == {
        "lexical": 12,
        "paraphrase": 12,
        "distractor": 12,
        "multi": 12,
    }
    assert Counter(passage.domain for passage in passages) == _DOMAINS
    assert len({passage.doc_id for passage in passages}) == 72
    assert len({query.query_id for query in queries}) == 48
    assert len({passage.text for passage in passages}) == 72
    assert len({query.text for query in queries}) == 48
    assert {passage.doc_id for passage in passages}.isdisjoint(query.query_id for query in queries)
    grades = {grade for query in queries for grade in query.relevance.values()}
    assert grades == {1, 2}
    assert ALLOWED_GRADES == frozenset({1, 2})
    assert EVAL_CATEGORIES == frozenset({"lexical", "paraphrase", "distractor", "multi"})
    partials = [grade for query in queries for grade in query.relevance.values() if grade == 1]
    assert len(partials) >= 6
    known = {passage.doc_id for passage in passages}
    blob = "\n".join([passage.text for passage in passages] + [query.text for query in queries])
    assert "http://" not in blob and "https://" not in blob
    assert "API_KEY" not in blob and "sk-" not in blob
    for passage in passages:
        size = len(tokenize(passage.text))
        assert 8 <= size <= 40, (passage.doc_id, size)
    for query in queries:
        assert query.category in EVAL_CATEGORIES
        assert set(query.relevance) <= known
        assert 2 in query.relevance.values()
        assert all(grade in {1, 2} for grade in query.relevance.values())


def test_v1_categories_separate_keyword_overlap_from_the_answer() -> None:
    passages = {passage.doc_id: passage for passage in load_eval_corpus()}
    for query in load_eval_queries():
        full = [doc_id for doc_id, grade in query.relevance.items() if grade == 2]
        if query.category == "multi":
            assert len(full) == 2, query.query_id
            left = _content(passages[full[0]].text)
            right = _content(passages[full[1]].text)
            asked = _content(query.text)
            only_left = {token for token in asked if token in left and token not in right}
            only_right = {token for token in asked if token in right and token not in left}
            assert only_left and only_right, (query.query_id, sorted(only_left), sorted(only_right))
            continue
        assert len(full) == 1, query.query_id
        answer = passages[full[0]].text
        if query.category == "paraphrase":
            assert _coverage(query.text, answer) <= 0.25, (
                query.query_id,
                _coverage(query.text, answer),
                sorted(_content(query.text) & _content(answer)),
            )
        elif query.category == "lexical":
            assert _coverage(query.text, answer) >= 0.8, (
                query.query_id,
                _coverage(query.text, answer),
            )
        elif query.category == "distractor":
            asked = _content(query.text)
            assert len(asked & _content(answer)) >= 2, query.query_id
            labeled = set(query.relevance)
            competitors = [
                passage.doc_id
                for passage in passages.values()
                if passage.doc_id not in labeled and len(asked & _content(passage.text)) >= 2
            ]
            assert len(competitors) >= 3, (query.query_id, competitors)
        else:
            raise AssertionError(query.category)


def test_v1_near_duplicates_change_a_fact_and_stay_unlabeled() -> None:
    passages = load_eval_corpus()
    by_id = {passage.doc_id: passage for passage in passages}
    links = [passage for passage in passages if passage.near_duplicate_of is not None]
    assert len(links) == 12
    for passage in links:
        canonical = by_id[passage.near_duplicate_of]
        assert canonical.near_duplicate_of is None
        assert passage.domain == canonical.domain
        assert passage.text != canonical.text
        score = _jaccard(passage.text, canonical.text)
        assert score >= 0.65, (passage.doc_id, round(score, 3))
    labeled: set[str] = set()
    for query in load_eval_queries():
        labeled.update(query.relevance)
        full = {doc_id for doc_id, grade in query.relevance.items() if grade == 2}
        for passage in links:
            assert passage.doc_id not in full
    assert labeled.isdisjoint(passage.doc_id for passage in links)


def test_readme_documents_v1_counts_without_retrieval_scores() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    section = readme.split("## Conjunto de avaliação", 1)[1].split("\n## ", 1)[0]
    assert "72 trechos" in section
    assert "48 perguntas" in section
    assert "12 de correspondência lexical" in section
    assert "12 de paráfrase" in section
    assert "12 com distratores" in section
    assert "12 multi-passagem" in section
    assert "12 quase-duplicatas" in section
    assert "Como foi construído" in section
    assert "Limitações" in section
    assert "Vila Serpa" in section
    assert "Relojoaria Lume" in section
    assert "Ribeirão Lanoso" in section
    assert "Bondinho" in section
    lowered = section.lower()
    assert "ndcg" not in lowered
    assert "mrr" not in lowered
    assert "recall" not in lowered


def test_package_exports_the_loader() -> None:
    import reranking_study as package

    assert package.load_eval_corpus is load_eval_corpus
    assert package.load_eval_queries is load_eval_queries


def _passage(doc_id: str, text: str = "Texto sintético da biblioteca municipal.") -> dict[str, str]:
    return {"id": doc_id, "domain": "biblioteca-vila-serpa", "text": text}


def _query(
    query_id: str,
    relevance: dict[str, int],
    category: str = "lexical",
    text: str = "Pergunta sintética sobre a biblioteca.",
) -> dict[str, object]:
    return {"id": query_id, "category": category, "text": text, "relevance": relevance}


def _write_corpus(directory: Path, rows: list[dict[str, object]]) -> None:
    _write_jsonl(directory / "passages.jsonl", rows)


def _write_queries(directory: Path, rows: list[dict[str, object]]) -> None:
    _write_jsonl(directory / "queries.jsonl", rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_loader_reads_a_minimal_directory_and_a_forward_near_duplicate(tmp_path: Path) -> None:
    _write_corpus(
        tmp_path,
        [
            {
                **_passage("p-dois", "Variante sintética do texto da biblioteca."),
                "near_duplicate_of": "p-um",
            },
            _passage("p-um"),
        ],
    )
    _write_queries(
        tmp_path,
        [_query("q-um", {"p-um": 2, "p-dois": 1}, category="distractor")],
    )
    passages = load_eval_corpus(tmp_path)
    queries = load_eval_queries(tmp_path)
    assert [passage.doc_id for passage in passages] == ["p-dois", "p-um"]
    assert passages[0].near_duplicate_of == "p-um"
    assert passages[1].near_duplicate_of is None
    assert queries[0].category == "distractor"
    assert dict(queries[0].relevance) == {"p-um": 2, "p-dois": 1}
    with pytest.raises(TypeError):
        queries[0].relevance["p-um"] = 1


def test_loader_rejects_duplicate_ids_unknown_labels_and_missing_grade_2(tmp_path: Path) -> None:
    _write_corpus(tmp_path, [_passage("p-um"), _passage("p-um")])
    with pytest.raises(ValueError, match="duplicate passage id"):
        load_eval_corpus(tmp_path)

    _write_corpus(tmp_path, [_passage("p-um"), _passage("p-dois", "Outro texto sintético.")])
    _write_queries(tmp_path, [_query("q-um", {"p-um": 2}), _query("q-um", {"p-dois": 2})])
    with pytest.raises(ValueError, match="duplicate query id"):
        load_eval_queries(tmp_path)

    _write_queries(tmp_path, [_query("q-um", {"p-ausente": 2})])
    with pytest.raises(ValueError, match="unknown passage id"):
        load_eval_queries(tmp_path)

    _write_queries(tmp_path, [_query("q-um", {"p-um": 1})])
    with pytest.raises(ValueError, match="grade 2"):
        load_eval_queries(tmp_path)


@pytest.mark.parametrize("grade", [0, 3, True, 1.0, "2"])
def test_loader_rejects_grades_outside_the_set(tmp_path: Path, grade: object) -> None:
    _write_corpus(tmp_path, [_passage("p-um")])
    _write_queries(tmp_path, [_query("q-um", {"p-um": grade})])  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="grade"):
        load_eval_queries(tmp_path)


@pytest.mark.parametrize("category", ["Lexical", "keyword", ""])
def test_loader_rejects_categories_outside_the_set(tmp_path: Path, category: str) -> None:
    _write_corpus(tmp_path, [_passage("p-um")])
    _write_queries(tmp_path, [_query("q-um", {"p-um": 2}, category=category)])
    with pytest.raises(ValueError, match="category"):
        load_eval_queries(tmp_path)


def test_loader_rejects_malformed_files_and_broken_near_duplicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval directory not found"):
        load_eval_corpus(tmp_path / "ausente")

    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="missing eval file"):
        load_eval_corpus(tmp_path)

    (tmp_path / "passages.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="eval corpus is empty"):
        load_eval_corpus(tmp_path)

    (tmp_path / "passages.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_eval_corpus(tmp_path)

    (tmp_path / "passages.jsonl").write_text(
        json.dumps(_passage("p-um"), ensure_ascii=False) + "\n\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty line"):
        load_eval_corpus(tmp_path)

    (tmp_path / "passages.jsonl").write_text(
        '{"id": "p-um", "id": "p-dois", "domain": "biblioteca-vila-serpa", '
        '"text": "Texto sintético da biblioteca."}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_eval_corpus(tmp_path)

    _write_corpus(tmp_path, [{**_passage("p-um"), "note": "sobra"}])
    with pytest.raises(ValueError, match="unknown passage field"):
        load_eval_corpus(tmp_path)

    _write_corpus(tmp_path, [_passage("p-um", "   ")])
    with pytest.raises(ValueError, match="non-empty"):
        load_eval_corpus(tmp_path)

    _write_corpus(
        tmp_path,
        [{**_passage("p-um"), "near_duplicate_of": "p-ausente"}],
    )
    with pytest.raises(ValueError, match="unknown passage id"):
        load_eval_corpus(tmp_path)

    _write_corpus(tmp_path, [{**_passage("p-um"), "near_duplicate_of": "p-um"}])
    with pytest.raises(ValueError, match="cannot point at itself"):
        load_eval_corpus(tmp_path)

    _write_corpus(
        tmp_path,
        [
            _passage("p-um"),
            {
                **_passage("p-dois", "Variante sintética do texto da biblioteca."),
                "near_duplicate_of": "p-um",
            },
            {
                **_passage("p-tres", "Outra variante sintética do mesmo texto."),
                "near_duplicate_of": "p-dois",
            },
        ],
    )
    with pytest.raises(ValueError, match="cannot chain"):
        load_eval_corpus(tmp_path)

    _write_corpus(tmp_path, [_passage("p-um")])
    (tmp_path / "queries.jsonl").write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_eval_queries(tmp_path)

    _write_queries(tmp_path, [{"id": "q-um", "category": "lexical", "text": "Pergunta."}])
    with pytest.raises(ValueError, match="missing query field"):
        load_eval_queries(tmp_path)

    (tmp_path / "passages.jsonl").write_text(
        "\ufeff" + json.dumps(_passage("p-um")) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="BOM"):
        load_eval_corpus(tmp_path)
