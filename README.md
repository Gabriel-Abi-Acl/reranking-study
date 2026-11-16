# reranking-study

Estudo técnico que compara pipelines de recuperação para RAG. Complementa os estudos `rag-study` e `chunking-study`. O autor do estudo é Gabriel, desenvolvedor de IA e automação no Brasil.

Este repositório mede qualidade e custo de quatro arranjos sobre o mesmo conjunto de perguntas. Não há resultado experimental neste repositório. Números só entram quando um experimento for executado e registrado.

## Objetivo

Comparar, no mesmo corpus e nas mesmas perguntas:

1. recuperação densa apenas;
2. recuperação lexical BM25 apenas;
3. híbrido denso + BM25 com Reciprocal Rank Fusion (RRF);
4. recuperação de primeiro estágio seguida de reranker (cross-encoder ou LLM-as-reranker) sobre o top-N.

Métricas previstas: recall@k, MRR, nDCG@k, latência por estágio e custo por consulta. Latência já é registrada pelo pipeline. Custo por consulta depende de chamada real e fica para o PR4.

## Hipótese

A hipótese abaixo será testada nos experimentos. Não é conclusão.

- O híbrido com RRF recupera mais documentos relevantes do que cada retriever isolado quando a consulta mistura termo literal e paráfrase.
- Um reranker sobre o top-N do primeiro estágio aumenta MRR e nDCG@k em relação ao ranking inicial, com latência e custo maiores por consulta.
- BM25 sozinho permanece competitivo quando a consulta repete o vocabulário do documento.

## Metodologia planejada

- Corpus e perguntas usam só texto sintético ou público. Nenhum documento privado entra no índice.
- Os quatro pipelines compartilham o índice em memória e o gabarito de relevância graduada.
- O primeiro estágio devolve candidatos. Se há reranker, ele reordena apenas o top-N e o pipeline devolve top-k. Se k passa de N, o restante acompanha a ordem do primeiro estágio.
- O embedder denso atual é um hash determinístico (SHA-256 por token, vetor L2-normalizado). Existe para fixar o contrato e os testes. Embeddings de modelo real ficam para o PR4.
- BM25 é Okapi em Python puro. IDF no estilo Lucene, `ln(1 + (N - df + 0.5) / (df + 0.5))`, com `k1 = 1.5` e `b = 0.75`. Cada termo distinto da consulta contribui uma vez.
- RRF usa `1 / (k + posição)` com `k = 60` e posição começando em 1. A fusão ignora a escala dos escores de origem.
- recall@k e MRR tratam grau maior que 0 como relevante. nDCG@k usa ganho `2^grau - 1` e desconto `log2(posição + 1)`.
- Configuração sensível vem de variável de ambiente. Chaves ficam fora do git.

O módulo `src/reranking_study/corpus.py` segue como placeholder de poucas frases para o harness. O conjunto da comparação é o v1 descrito abaixo. Nenhum dos dois sustenta conclusão sozinho.

## Conjunto de avaliação (v1)

O v1 tem 72 trechos e 48 perguntas. As categorias ficam empatadas: 12 de correspondência lexical, 12 de paráfrase, 12 com distratores e 12 multi-passagem. Há 12 quase-duplicatas. Os trechos se dividem em quatro domínios fictícios, 18 em cada: a Biblioteca Municipal de Vila Serpa, a Relojoaria Lume, a Cooperativa Agrícola Ribeirão Lanoso e o Museu do Bondinho de Vila Serpa. Vila Serpa, o Ribeirão Lanoso, a relojoaria e as pessoas citadas são inventados.

Os arquivos são `data/eval/v1/passages.jsonl` e `data/eval/v1/queries.jsonl`. `load_eval_corpus` e `load_eval_queries` recusam id repetido, rótulo apontando para trecho inexistente, pergunta sem grau 2, grau fora de {1, 2} e categoria fora do conjunto fixo (`lexical`, `paraphrase`, `distractor`, `multi`). O grau 0 não é gravado: trecho ausente no mapa da pergunta é irrelevante.

### Como foi construído

O texto foi escrito à mão para este estudo, em português, em frases curtas. Cada quase-duplicata repete um trecho e troca um fato (horário, preço, prazo, lugar ou uma negação); o campo `near_duplicate_of` liga a variante ao original. Os distratores reaproveitam o nome da instituição ou outra palavra da pergunta e respondem um fato diferente. As paráfrases foram redigidas com outro vocabulário, de propósito, para o enunciado não repetir as palavras do trecho. Nas perguntas multi-passagem a resposta foi partida em dois trechos: cada um guarda uma parte necessária e os dois recebem grau 2. O grau 1 marca contexto verdadeiro do mesmo assunto que não fecha a resposta. Um único anotador atribuiu os graus.

### Limitações

O conjunto é pequeno, de um idioma e de um anotador só, sem segunda revisão. As categorias saem da redação do texto. Quase-duplicata e distrator podem ser fáceis para uma pessoa e ainda assim trocar a ordem de um ranking por palavras; isso fica para o experimento. As paráfrases controlam a sobreposição de palavras na redação, sem um modelo de sinônimos. O v1 fixa o gabarito. Ele não é um relatório de recuperação.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Extras opcionais. Os testes não instalam nem chamam esses pacotes:

```bash
uv sync --extra cross-encoder
uv sync --extra llm
```

Copie `.env.example` para `.env` e exporte as variáveis no ambiente local. O exemplo versionado deixa os valores vazios. Não commite `.env` nem chave de API.

```bash
export OPENAI_API_KEY=
export OPENAI_BASE_URL=
export OPENAI_MODEL=
export CROSS_ENCODER_MODEL=
```

Uso mínimo do contrato, com o reranker determinístico:

```python
from reranking_study import Bm25Retriever, Document, FakeReranker, RetrievalPipeline, recall_at_k

documents = [Document("doc-bm25", "BM25 pontua termos raros em documentos curtos.")]
retriever = Bm25Retriever()
retriever.index(documents)
pipeline = RetrievalPipeline(retriever, reranker=FakeReranker(), rerank_top_n=5)
result = pipeline.run("BM25", k=1)
print(result.ids, dict(result.stage_latency_ms))
print(recall_at_k(result.ids, {"doc-bm25": 1}, k=1))
```

`CrossEncoderReranker` e `LlmReranker` importam `sentence-transformers` e `openai` só na primeira chamada de `rerank`. Sem o extra, ou sem `OPENAI_API_KEY` no caso do LLM, a chamada falha em vez de devolver escore.

## O que o repositório já cobre

Contrato do pipeline, índice em memória, métricas por consulta, retrievers e reranker determinísticos, e o conjunto `data/eval/v1`. CI roda lint e testes. Embeddings de modelo, cross-encoder real, LLM e o relatório de custo ficam para os PRs seguintes. Este texto não traz número de experimento.

## Roadmap

- PR2: corpus e perguntas com gabarito em `data/eval/v1`.
- PR3: avaliação — agregação de recall@k, MRR, nDCG@k e o registro de latência por estágio.
- PR4: experimentos com embeddings, cross-encoder e LLM-as-reranker, mais o relatório de métricas, latência e custo por consulta. Sem resultado inventado.
