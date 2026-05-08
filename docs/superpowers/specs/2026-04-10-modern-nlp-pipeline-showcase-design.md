# Modern NLP Pipeline Showcase Design

## Scope

Build a new showcase project under `projects/modern-nlp-pipeline-showcase` that teaches one coherent end-to-end NLP workflow on a shared corpus of research abstracts and paper summaries.

The workflow should cover:

1. corpus ingestion and validation,
2. lexical and dense text representation,
3. topic classification,
4. semantic retrieval with lexical and dense baselines,
5. retrieval-grounded QA and query-focused summarization,
6. artifact verification and student-facing interpretation.

## Primary Learning Outcome

Learners should be able to explain how one modern NLP pipeline reuses the same corpus and representations across discriminative, retrieval, and generative tasks.

## Audience

- intermediate students,
- comfortable with Python and basic machine learning,
- new to retrieval and transformer-era NLP systems.

## Constraints

- normal laptop friendly,
- CPU runnable by default,
- no external services,
- script-first execution,
- deterministic artifact paths,
- local artifact verifier,
- public-safe and open-source friendly.

## Design Decisions

### 1. Shared corpus instead of three disconnected demos

The showcase should not be a broad survey of unrelated NLP tasks. It should keep one corpus and one narrative:

- papers belong to topic labels for classification,
- the same papers are chunked for retrieval,
- retrieved chunks supply evidence for QA and summarization.

This satisfies the repo playbook requirement for one coherent workflow.

### 2. Lexical baseline plus compact dense models

The project should keep a strong baseline and a modern main path:

- lexical baseline: `TF-IDF`,
- dense encoder path: a sentence-transformer compatible compact encoder,
- optional reranker: a compact cross-encoder only when the dependency budget stays reasonable.

The teaching value comes from comparing sparse and dense retrieval behavior, not from chasing the absolute largest model.

### 3. Student-friendly generation path

Generation should stay grounded and auditable:

- QA should operate over retrieved passages,
- summarization should use retrieved evidence and expose the supporting passages,
- outputs should be written to stable CSV/JSON/Markdown artifacts.

### 4. Small curated local dataset

The project should ship with a local dataset of research abstracts and short paper summaries so the first run is reproducible and does not depend on external dataset downloads.

## Proposed Project Structure

```text
projects/modern-nlp-pipeline-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture-notes.md
│   ├── learning-guide.md
│   └── model-notes.md
├── scripts/
│   ├── run_pipeline.py
│   └── verify_artifacts.py
├── src/modern_nlp_pipeline_showcase/
│   ├── __init__.py
│   ├── classification.py
│   ├── config.py
│   ├── data.py
│   ├── generation.py
│   ├── lexical.py
│   ├── models.py
│   ├── reporting.py
│   └── retrieval.py
├── tests/
└── artifacts/
```

## Artifact Contract

Required files:

- `artifacts/manifest.json`
- `artifacts/data/corpus_overview.csv`
- `artifacts/data/topic_distribution.csv`
- `artifacts/classification/metrics_summary.csv`
- `artifacts/retrieval/retrieval_metrics.csv`
- `artifacts/retrieval/retrieval_examples.json`
- `artifacts/generation/qa_outputs.csv`
- `artifacts/generation/query_summaries.json`
- `artifacts/summary.md`

## Success Criteria

- `make run` generates the full artifact contract.
- `make verify` validates the artifact contract.
- `make check` passes in the project directory.
- Root-level docs, CI, and issue templates include the new showcase.
- The README explains how to interpret each artifact in plain language.
