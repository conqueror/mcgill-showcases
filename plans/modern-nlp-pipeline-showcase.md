# Modern NLP Pipeline Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Build `projects/modern-nlp-pipeline-showcase` as a student-friendly, end-to-end NLP project that uses one corpus of research abstracts and paper summaries for classification, retrieval, and retrieval-grounded generation.

**Architecture:** The showcase will use a local curated corpus plus a script-first pipeline. The implementation will compare a lexical baseline with dense sentence embeddings, then reuse the same data products across topic classification, semantic retrieval, QA, and query-focused summarization. Artifacts will be written to stable paths and verified by a local contract checker.

**Reasoning:** This structure keeps the repo-compliant “one coherent workflow” requirement while still covering the broader NLP landscape the user requested. A shared corpus makes it easy for students to compare how the same text behaves under sparse, dense, discriminative, and retrieval-grounded generative settings.

**Tech Stack:** Python 3.11, uv, pandas, scikit-learn, numpy, sentence-transformers, transformers, torch, pytest, ruff, mypy

---

## Scope

In scope:

- scaffold a new project under `projects/modern-nlp-pipeline-showcase`,
- ship a small local corpus of research abstracts and paper summaries,
- implement lexical and dense representation paths,
- implement topic classification, semantic retrieval, QA, and query-focused summarization,
- generate stable artifacts and a verifier,
- wire the project into root Makefile, CI, docs, and issue templates.

Out of scope:

- large benchmark datasets,
- GPU-only workflows,
- serving APIs,
- notebook-heavy teaching paths,
- claim of absolute best leaderboard performance.

## Assumptions

- The design spec at `docs/superpowers/specs/2026-04-10-modern-nlp-pipeline-showcase-design.md` is the active design reference.
- Compact open models are acceptable when they keep the default path CPU-friendly.
- The first run may download model weights from Hugging Face unless already cached.
- Tests should avoid network downloads and validate logic with small fixtures.

## Stop Conditions

Stop and re-scope if:

- the project is pushed beyond one coherent workflow,
- the required models become too heavy for a normal laptop default path,
- runtime requires a GPU,
- root integration requirements expand beyond the public-safe repo playbook.

## Success Criteria

- `projects/modern-nlp-pipeline-showcase` exists with the standard showcase structure.
- `cd projects/modern-nlp-pipeline-showcase && make run` generates all required artifacts.
- `cd projects/modern-nlp-pipeline-showcase && make verify` passes.
- `cd projects/modern-nlp-pipeline-showcase && make check` passes.
- Root `README.md`, `docs/getting-started.md`, `docs/learning-path.md`, `docs/aspect-coverage-matrix.md`, `.github/workflows/ci.yml`, and issue templates include the new showcase.

## Task 1: Scaffold the project and seed the local corpus

**Files:**
- Create: `projects/modern-nlp-pipeline-showcase/README.md`
- Create: `projects/modern-nlp-pipeline-showcase/Makefile`
- Create: `projects/modern-nlp-pipeline-showcase/pyproject.toml`
- Create: `projects/modern-nlp-pipeline-showcase/data/raw/research_corpus.csv`
- Create: `projects/modern-nlp-pipeline-showcase/data/raw/research_queries.json`
- Create: `projects/modern-nlp-pipeline-showcase/artifacts/.gitkeep`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/__init__.py`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/config.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_package_imports.py`

**Step 1: Write the failing test**

- Add `tests/test_package_imports.py` to assert the package imports and exposes a version string.

**Step 2: Run test to verify it fails**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_package_imports.py -q`

**Step 3: Write minimal implementation**

- Add project scaffolding, package metadata, and the local corpus/query fixtures.

**Step 4: Run test to verify it passes**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_package_imports.py -q`

## Task 2: Implement corpus loading, chunking, and lexical features

**Files:**
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/data.py`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/lexical.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_data.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_lexical.py`

**Step 1: Write the failing tests**

- Add tests for corpus schema validation, chunk generation, and lexical vectorizer output shapes.

**Step 2: Run tests to verify they fail**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_data.py tests/test_lexical.py -q`

**Step 3: Write minimal implementation**

- Implement corpus readers, chunk builders, label preparation, TF-IDF fitting, and lexical retrieval scoring.

**Step 4: Run tests to verify they pass**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_data.py tests/test_lexical.py -q`

## Task 3: Implement dense models, classification, retrieval, and generation

**Files:**
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/models.py`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/classification.py`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/retrieval.py`
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/generation.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_classification.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_retrieval.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_generation.py`

**Step 1: Write the failing tests**

- Add tests for embedding backend selection, classifier metrics schema, retrieval metric calculation, and retrieval-grounded output formatting.

**Step 2: Run tests to verify they fail**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_classification.py tests/test_retrieval.py tests/test_generation.py -q`

**Step 3: Write minimal implementation**

- Implement a compact embedding backend, dense-vs-lexical comparison utilities, classifier training/evaluation, retrieval evaluation, extractive QA, and query-focused summarization.

**Step 4: Run tests to verify they pass**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_classification.py tests/test_retrieval.py tests/test_generation.py -q`

## Task 4: Implement the pipeline entrypoint, artifact reporting, and contract verification

**Files:**
- Create: `projects/modern-nlp-pipeline-showcase/src/modern_nlp_pipeline_showcase/reporting.py`
- Create: `projects/modern-nlp-pipeline-showcase/scripts/run_pipeline.py`
- Create: `projects/modern-nlp-pipeline-showcase/scripts/verify_artifacts.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_pipeline.py`
- Create: `projects/modern-nlp-pipeline-showcase/tests/test_verify_artifacts.py`

**Step 1: Write the failing tests**

- Add tests for manifest contents, report-writing paths, and pipeline smoke execution in quick mode.

**Step 2: Run tests to verify they fail**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_pipeline.py tests/test_verify_artifacts.py -q`

**Step 3: Write minimal implementation**

- Implement the script entrypoint, artifact writers, summary Markdown, and verifier.

**Step 4: Run tests to verify they pass**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_pipeline.py tests/test_verify_artifacts.py -q`

## Task 5: Write the student-facing docs

**Files:**
- Create: `projects/modern-nlp-pipeline-showcase/docs/architecture-notes.md`
- Create: `projects/modern-nlp-pipeline-showcase/docs/learning-guide.md`
- Create: `projects/modern-nlp-pipeline-showcase/docs/model-notes.md`

**Step 1: Write the failing test**

- Extend `tests/test_pipeline.py` or add a docs manifest test to require the expected README sections and docs files.

**Step 2: Run test to verify it fails**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_pipeline.py -q`

**Step 3: Write minimal implementation**

- Document the project purpose, prerequisites, quickstart, artifact interpretation, failure modes, and next projects.

**Step 4: Run test to verify it passes**

- Command: `cd projects/modern-nlp-pipeline-showcase && uv run pytest tests/test_pipeline.py -q`

## Task 6: Add root integrations and final verification

**Files:**
- Modify: `Makefile`
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/learning-path.md`
- Modify: `docs/aspect-coverage-matrix.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Modify: `.github/ISSUE_TEMPLATE/learning-question.yml`
- Modify: `.github/ISSUE_TEMPLATE/feature_request.yml`

**Step 1: Write the failing test**

- Use project-local tests plus docs checks and root grep validation as the integration failure signal.

**Step 2: Run verification to observe the missing integration state**

- Commands:
  - `cd projects/modern-nlp-pipeline-showcase && uv run pytest -q`
  - `cd /Users/fatih/dev/mcgill-showcases && rg -n "modern-nlp-pipeline-showcase" README.md docs .github/workflows/ci.yml .github/ISSUE_TEMPLATE`

**Step 3: Write minimal implementation**

- Add the new showcase to root orchestration, docs, CI, and issue templates.

**Step 4: Run final verification**

- Commands:
  - `cd projects/modern-nlp-pipeline-showcase && make run`
  - `cd projects/modern-nlp-pipeline-showcase && make verify`
  - `cd projects/modern-nlp-pipeline-showcase && make check`
  - `cd /Users/fatih/dev/mcgill-showcases && make docs-check`

