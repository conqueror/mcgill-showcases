# Modern NLP Pipeline Showcase

Learn one coherent modern NLP workflow on a shared corpus of research abstracts and paper summaries.

This project keeps the same text corpus and reuses it across three downstream tasks:

1. topic classification,
2. semantic retrieval,
3. retrieval-grounded question answering and query-focused summarization.

The main teaching goal is to show how sparse baselines and dense transformer-era representations behave differently on the same documents, queries, and artifact contract.

## What You Should Learn

By working through this showcase, you should be able to:

- explain why `TF-IDF` is a strong lexical baseline but not the same thing as tokenization,
- compare sparse lexical features with dense sentence embeddings,
- evaluate topic classification with interpretable metrics,
- evaluate retrieval with recall@k and MRR@k,
- explain why generation quality depends on retrieval quality,
- read the artifact outputs and summarize them in plain language.

## Prerequisites

- Python 3.11+
- `uv`
- basic Python and pandas familiarity
- basic machine learning intuition

Notes:

- `make run` may download compact Hugging Face model weights the first time.
- `make smoke` uses lightweight fallback backends so the project remains runnable even when model downloads are unavailable.

## Quickstart

```bash
cd projects/modern-nlp-pipeline-showcase
make sync
make smoke
make verify
make test
```

Run the fuller pipeline:

```bash
make run
```

## Key Artifacts

- `artifacts/data/corpus_overview.csv`
  - one row per paper with topic label and text-length summary
- `artifacts/data/topic_distribution.csv`
  - class balance across research topics
- `artifacts/classification/metrics_summary.csv`
  - topic classification metrics for lexical and dense baselines
- `artifacts/retrieval/retrieval_metrics.csv`
  - retrieval quality comparison across lexical and dense strategies
- `artifacts/retrieval/retrieval_examples.json`
  - per-query examples showing which papers and chunks were retrieved
- `artifacts/generation/qa_outputs.csv`
  - retrieval-grounded answers to research questions
- `artifacts/generation/query_summaries.json`
  - short query-focused summaries from retrieved evidence
- `artifacts/summary.md`
  - one-page interpretation of the run
- `artifacts/manifest.json`
  - required artifact contract for verification

## How To Learn This Project

1. Start with `docs/learning-guide.md`.
2. Run `make smoke` to see the whole flow quickly.
3. Open `artifacts/classification/metrics_summary.csv` and compare lexical vs dense classifiers.
4. Open `artifacts/retrieval/retrieval_metrics.csv` and check whether dense retrieval improves semantic match quality.
5. Open `artifacts/retrieval/retrieval_examples.json` and inspect what each strategy surfaces.
6. Open `artifacts/generation/qa_outputs.csv` and confirm that answers remain tied to retrieved evidence.
7. Read `docs/model-notes.md` to understand the dense encoder and fallback choices.

## Makefile Commands

```bash
make help
make sync
make run
make smoke
make verify
make test
make ruff
make ty
make check
```

## Common Failure Modes

- The first full run takes longer than expected:
  The dense encoder or generation models may be downloading for the first time. Use `make smoke` first.
- `make verify` fails:
  Run `make run` or `make smoke` first so the required artifacts exist.
- Dense retrieval does not beat lexical retrieval on every query:
  That is normal. The point of the project is to inspect when semantic representations help and when exact terminology still matters.
- Generated answers look generic:
  Check `artifacts/retrieval/retrieval_examples.json` first. Weak retrieval usually leads to weak grounded generation.

## Suggested Next Projects

- `projects/learning-to-rank-foundations-showcase`
- `projects/autoresearch`
- `projects/ranking-api-productization-showcase`

## Project Structure

```text
modern-nlp-pipeline-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── data/
├── docs/
├── scripts/
├── src/modern_nlp_pipeline_showcase/
├── tests/
└── artifacts/
```
