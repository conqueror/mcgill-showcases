# Deep Learning Math Foundations Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Build `projects/deep-learning-math-foundations-showcase` as a self-guided, runnable project that teaches the essential math behind deep learning through small Python modules, strong docs, and inspectable artifacts.

**Architecture:** The project will follow the repo's standalone showcase pattern: a single project folder with `README.md`, `Makefile`, `pyproject.toml`, `src/`, `tests/`, `scripts/`, `docs/`, `data/`, and `artifacts/`. The code path stays script-first and artifact-driven, while the docs map each concept to runnable outputs and checkpoint questions.

**Reasoning:** This is the best Phase 1 project because it has the least framework overhead and establishes the documentation and artifact conventions that later deep learning showcases can reuse. The implementation should stay intentionally small and deterministic so the learning experience is clear and the code remains easy to read, test, and explain.

**Tech Stack:** Python 3.11, uv, NumPy, pandas, Matplotlib, SymPy, pytest, Ruff, ty, Markdown

---

## Scope

In scope:

- scaffold a new standalone showcase project under `projects/`,
- implement math teaching modules for linear algebra, calculus, probability, information theory, and optimization,
- generate inspectable artifacts from a single script entrypoint,
- provide beginner-friendly docs and concept-to-artifact learning flow,
- add tests, lint/type-check wiring, and artifact verification.

Out of scope:

- notebooks in the first pass,
- external datasets,
- PyTorch or TensorFlow usage,
- neural network model training,
- advanced math beyond what directly supports later deep learning showcases.

## Assumptions

- The approved design spec at `docs/superpowers/specs/2026-03-13-deep-learning-showcase-series-design.md` remains the governing design document.
- Phase 1 covers only `projects/deep-learning-math-foundations-showcase`.
- SymPy is acceptable for lightweight symbolic examples; if it becomes a maintenance burden, replace with numeric demonstrations only.
- The project should follow the command style already used in other showcases: `make sync`, `make run`, `make test`, `make quality`.

## Stop Conditions

Stop execution and ask for direction if:

- the project split changes from the approved three-project design,
- the user wants notebooks included in the initial pass,
- the user wants Keras, TensorFlow, or PyTorch in this math foundations project,
- the repo has a stronger project template that conflicts with the assumed structure,
- dependency choices need to change materially for CI or portability reasons.

## Success Criteria

- `projects/deep-learning-math-foundations-showcase` exists with the expected showcase layout.
- `uv run pytest` passes inside the project.
- `uv run python scripts/run_showcase.py` generates the expected artifacts.
- `uv run python scripts/verify_artifacts.py` exits successfully.
- The docs give a clean self-guided path from math concept to code to artifact.
- Public modules, functions, and classes have useful docstrings.

## Task 1: Scaffold the project skeleton and package import path

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/README.md`
- Create: `projects/deep-learning-math-foundations-showcase/Makefile`
- Create: `projects/deep-learning-math-foundations-showcase/pyproject.toml`
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/__init__.py`
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/config.py`
- Create: `projects/deep-learning-math-foundations-showcase/scripts/run_showcase.py`
- Create: `projects/deep-learning-math-foundations-showcase/scripts/verify_artifacts.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_package_imports.py`
- Create: `projects/deep-learning-math-foundations-showcase/data/raw/.gitkeep`
- Create: `projects/deep-learning-math-foundations-showcase/data/processed/.gitkeep`
- Create: `projects/deep-learning-math-foundations-showcase/artifacts/.gitkeep`

**Step 1: Write the failing test**

- Add `tests/test_package_imports.py` that imports `deep_learning_math_foundations_showcase` and checks a version or project constant is exposed.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_package_imports.py -q`
- Expected: failure due to missing package files and project config.

**Step 3: Write minimal implementation**

- Create the project folder structure, basic package files, `pyproject.toml`, and minimal `Makefile` targets for `sync`, `run`, `test`, `ruff-check`, `ruff-format`, `ty-check`, `lint`, and `quality`.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_package_imports.py -q`
- Expected: test passes and package imports cleanly.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 2: Implement linear algebra teaching utilities

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/linear_algebra.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_linear_algebra.py`

**Step 1: Write the failing test**

- Add tests for vector addition, scalar multiplication, dot product, and a simple 2D matrix transform with deterministic outputs.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_linear_algebra.py -q`
- Expected: failure because `linear_algebra.py` does not exist.

**Step 3: Write minimal implementation**

- Implement small, readable helpers that return numeric results or DataFrames used later by reporting code.
- Add docstrings that explain the teaching purpose of each helper.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_linear_algebra.py -q`
- Expected: passing deterministic math checks.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 3: Implement calculus teaching utilities

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/calculus.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_calculus.py`

**Step 1: Write the failing test**

- Add tests for derivative examples, partial derivative examples, and a simple finite-difference or symbolic check that stays deterministic.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_calculus.py -q`
- Expected: failure because the module is missing.

**Step 3: Write minimal implementation**

- Implement small utilities for derivative demonstrations and structured outputs that can become artifact tables.
- Keep symbolic usage limited and easy to read.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_calculus.py -q`
- Expected: passing deterministic calculus checks.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 4: Implement probability and statistics teaching utilities

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/probability.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_probability.py`

**Step 1: Write the failing test**

- Add tests for sample mean/variance summaries, Bernoulli simulation summaries, and uncertainty propagation outputs with fixed random seeds.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_probability.py -q`
- Expected: failure because the module is missing.

**Step 3: Write minimal implementation**

- Implement reproducible simulation helpers and summary builders that explain why the outputs matter for machine learning.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_probability.py -q`
- Expected: passing seeded simulation and summary checks.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 5: Implement information theory teaching utilities

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/information_theory.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_information_theory.py`

**Step 1: Write the failing test**

- Add tests for entropy, cross-entropy, and KL divergence on small deterministic distributions.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_information_theory.py -q`
- Expected: failure because the module is missing.

**Step 3: Write minimal implementation**

- Implement small numeric helpers and structured explanations that can later feed artifact tables and summary Markdown.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_information_theory.py -q`
- Expected: passing deterministic distribution checks.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 6: Implement optimization teaching utilities

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/optimization.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_optimization.py`

**Step 1: Write the failing test**

- Add tests for a simple gradient descent trace, monotonic loss reduction on a toy convex example, and stable trace schema.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_optimization.py -q`
- Expected: failure because the module is missing.

**Step 3: Write minimal implementation**

- Implement a tiny, transparent optimizer demo with seeded or deterministic starting conditions.
- Keep the code educational rather than generic or extensible.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_optimization.py -q`
- Expected: passing trace and convergence checks.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 7: Implement plotting and reporting helpers

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/plots.py`
- Create: `projects/deep-learning-math-foundations-showcase/src/deep_learning_math_foundations_showcase/reporting.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_reporting.py`

**Step 1: Write the failing test**

- Add tests for summary artifact schemas, expected Markdown section headers, and plot-output path planning without requiring visual inspection.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_reporting.py -q`
- Expected: failure because reporting helpers are missing.

**Step 3: Write minimal implementation**

- Implement helpers that convert module outputs into CSV and Markdown artifacts with stable field names.
- Keep plot generation optional and lightweight.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_reporting.py -q`
- Expected: artifact schema and summary-format checks pass.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 8: Wire the end-to-end showcase script

**Files:**
- Modify: `projects/deep-learning-math-foundations-showcase/scripts/run_showcase.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_pipeline.py`

**Step 1: Write the failing test**

- Add an integration test that runs the showcase in a temp artifact directory and asserts all expected artifact files are created.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_pipeline.py -q`
- Expected: failure because the pipeline script does not generate required outputs yet.

**Step 3: Write minimal implementation**

- Wire the concept modules into `scripts/run_showcase.py`.
- Generate the initial artifact set:
  - `artifacts/vector_operations.csv`
  - `artifacts/matrix_transformations.csv`
  - `artifacts/derivative_examples.csv`
  - `artifacts/gradient_descent_trace.csv`
  - `artifacts/probability_simulations.csv`
  - `artifacts/information_theory_summary.md`
  - `artifacts/summary.md`

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_pipeline.py -q`
- Expected: integration test passes and artifact filenames match the agreed schema.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 9: Add artifact verification

**Files:**
- Modify: `projects/deep-learning-math-foundations-showcase/scripts/verify_artifacts.py`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_verify_artifacts.py`

**Step 1: Write the failing test**

- Add tests that detect missing artifact files or malformed summary output.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_verify_artifacts.py -q`
- Expected: failure because artifact verification logic is not implemented yet.

**Step 3: Write minimal implementation**

- Implement artifact existence checks and simple content validation for required files.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_verify_artifacts.py -q`
- Expected: verification logic passes for good fixtures and fails for incomplete ones.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 10: Write the core self-guided docs set

**Files:**
- Modify: `projects/deep-learning-math-foundations-showcase/README.md`
- Create: `projects/deep-learning-math-foundations-showcase/docs/learning-flow.md`
- Create: `projects/deep-learning-math-foundations-showcase/docs/concept-learning-map.md`
- Create: `projects/deep-learning-math-foundations-showcase/docs/code-examples.md`
- Create: `projects/deep-learning-math-foundations-showcase/docs/domain-use-cases.md`
- Create: `projects/deep-learning-math-foundations-showcase/docs/checkpoint-answer-key.md`
- Create: `projects/deep-learning-math-foundations-showcase/tests/test_docs_manifest.py`

**Step 1: Write the failing test**

- Add `tests/test_docs_manifest.py` that asserts the required docs exist and contain the expected top-level sections.

**Step 2: Run test to verify it fails**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_docs_manifest.py -q`
- Expected: failure because the docs set is incomplete.

**Step 3: Write minimal implementation**

- Write docs that follow the standard learning unit pattern:
  - intuition,
  - minimal math,
  - code anchor,
  - expected output,
  - common mistake,
  - checkpoint question.
- Keep README focused on learning outcomes, quickstart, key outputs, and suggested study path.

**Step 4: Run test to verify it passes**

- Command: `cd projects/deep-learning-math-foundations-showcase && uv run pytest tests/test_docs_manifest.py -q`
- Expected: required docs exist and include the agreed sections.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 11: Add the project quality gate

**Files:**
- Create: `projects/deep-learning-math-foundations-showcase/scripts/run_quality_checks.sh`
- Modify: `projects/deep-learning-math-foundations-showcase/Makefile`

**Step 1: Write the failing test**

- No new pytest file is required; use command verification for this task because it is a shell-level quality gate.

**Step 2: Run verification to show the gap**

- Command: `cd projects/deep-learning-math-foundations-showcase && make quality`
- Expected: failure because the quality script is missing.

**Step 3: Write minimal implementation**

- Add `scripts/run_quality_checks.sh` and wire `make quality` to run:
  - `uv run ruff check src tests scripts`
  - `uv run ruff format --check src tests scripts`
  - `uv run ty check src tests scripts --ignore unresolved-import --ignore unresolved-attribute`
  - `uv run pytest`

**Step 4: Run verification to prove it works**

- Command: `cd projects/deep-learning-math-foundations-showcase && make quality`
- Expected: command exits successfully.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Task 12: Final project verification and evidence capture

**Files:**
- No new files required unless gaps are found during verification.

**Step 1: Write the failing test**

- No new test file; this task is the final execution and evidence pass.

**Step 2: Run verification to expose remaining gaps**

- Commands:
  - `cd projects/deep-learning-math-foundations-showcase && uv sync --extra dev`
  - `cd projects/deep-learning-math-foundations-showcase && uv run pytest`
  - `cd projects/deep-learning-math-foundations-showcase && uv run python scripts/run_showcase.py`
  - `cd projects/deep-learning-math-foundations-showcase && uv run python scripts/verify_artifacts.py`
  - `cd projects/deep-learning-math-foundations-showcase && make quality`

- Expected: one or more commands may still fail before cleanup.

**Step 3: Write minimal implementation**

- Fix only the remaining gaps surfaced by the full-project run.

**Step 4: Run verification to prove completion**

- Repeat the full command set above.
- Expected: all commands pass and the project produces the agreed artifacts.

**Step 5: Commit (only after approval)**

- Commit only after explicit user approval during execution.

## Completion Evidence

The implementation is complete when the following evidence exists:

- the new project folder is present at `projects/deep-learning-math-foundations-showcase`,
- required docs exist and are readable,
- required artifacts are generated and verified,
- test, lint, type-check, and project run commands all pass,
- the code is readable and public APIs have useful docstrings.

## Review Request

After this plan is approved, execution should start with **Task 1 only** using `core-executing-plans`, without broadening scope into Phase 2 or Phase 3.
