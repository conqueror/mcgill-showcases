# PyTorch Training Regularization Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Build `projects/pytorch-training-regularization-showcase` as a self-guided project that teaches tensors, autograd, `nn.Module`, `Dataset`/`DataLoader`, training loops, metrics, optimizers, schedulers, dropout, batch norm, early stopping, and regularization tradeoffs through a polished classifier workflow.

**Architecture:** The project will be a standalone showcase with a PyTorch-first package, script-based experiments, deterministic test doubles, and a student-facing pipeline centered on a FashionMNIST classifier plus lightweight optimizer and regularization comparisons. Tests will use synthetic tensor datasets so local quality gates do not depend on downloads.

**Reasoning:** This project is the framework-engineering step of the deep learning series. It should feel practical and real, but it must still remain legible and beginner-friendly, which means short modules, stable artifacts, and experiments that show one training or regularization decision at a time.

**Tech Stack:** Python 3.11, uv, PyTorch, torchvision, NumPy, pandas, Matplotlib, pytest, Ruff, ty, Markdown

---

### Task 1: Scaffold the project structure and package path

**Files:**
- Create: `projects/pytorch-training-regularization-showcase/README.md`
- Create: `projects/pytorch-training-regularization-showcase/Makefile`
- Create: `projects/pytorch-training-regularization-showcase/pyproject.toml`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/__init__.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/config.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/__init__.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/run_showcase.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/run_optimizer_comparison.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/run_regularization_ablation.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/verify_artifacts.py`
- Create: `projects/pytorch-training-regularization-showcase/scripts/run_quality_checks.sh`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_package_imports.py`
- Create: `projects/pytorch-training-regularization-showcase/artifacts/manifest.json`
- Create: `projects/pytorch-training-regularization-showcase/artifacts/.gitkeep`
- Create: `projects/pytorch-training-regularization-showcase/data/raw/.gitkeep`
- Create: `projects/pytorch-training-regularization-showcase/data/processed/.gitkeep`

**Step 1: Write the failing import test**

**Step 2: Run the missing-project failure**

**Step 3: Create the minimal scaffold**

**Step 4: Rerun the import test**

### Task 2: Implement data, model, and evaluation primitives

**Files:**
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/data.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/models.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/evaluation.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_data.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_models.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_evaluation.py`

**Step 1: Write failing tests for tensor datasets, model shapes, and metric helpers**

**Step 2: Run targeted failing tests**

**Step 3: Implement the primitives**

**Step 4: Rerun targeted tests**

### Task 3: Implement training, regularization, and experiment helpers

**Files:**
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/training.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/regularization.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/experiments.py`
- Create: `projects/pytorch-training-regularization-showcase/src/pytorch_training_regularization_showcase/reporting.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_training.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_regularization.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_experiments.py`

**Step 1: Write failing tests for training history, early stopping, optimizer comparison, and regularization summaries**

**Step 2: Run targeted failing tests**

**Step 3: Implement the training and experiment helpers**

**Step 4: Rerun targeted tests**

### Task 4: Wire the showcase scripts and artifact verification

**Files:**
- Modify: `projects/pytorch-training-regularization-showcase/scripts/run_showcase.py`
- Modify: `projects/pytorch-training-regularization-showcase/scripts/run_optimizer_comparison.py`
- Modify: `projects/pytorch-training-regularization-showcase/scripts/run_regularization_ablation.py`
- Modify: `projects/pytorch-training-regularization-showcase/scripts/verify_artifacts.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_pipeline.py`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_verify_artifacts.py`

**Step 1: Write failing pipeline and verifier tests**

**Step 2: Run targeted failing tests**

**Step 3: Implement artifact generation for:
  - `baseline_metrics.json`
  - `training_history.csv`
  - `optimizer_comparison.csv`
  - `learning_rate_schedule_comparison.csv`
  - `regularization_ablation.csv`
  - `gradient_health_report.md`
  - `error_analysis.csv`
  - `summary.md`
**

**Step 4: Rerun targeted tests**

### Task 5: Write the self-guided docs set

**Files:**
- Modify: `projects/pytorch-training-regularization-showcase/README.md`
- Create: `projects/pytorch-training-regularization-showcase/docs/learning-flow.md`
- Create: `projects/pytorch-training-regularization-showcase/docs/concept-learning-map.md`
- Create: `projects/pytorch-training-regularization-showcase/docs/code-examples.md`
- Create: `projects/pytorch-training-regularization-showcase/docs/domain-use-cases.md`
- Create: `projects/pytorch-training-regularization-showcase/docs/checkpoint-answer-key.md`
- Create: `projects/pytorch-training-regularization-showcase/tests/test_docs_manifest.py`

**Step 1: Write the failing docs manifest test**

**Step 2: Run the failing docs test**

**Step 3: Write the docs**

**Step 4: Rerun the docs test**

### Task 6: Run the full project verification loop

**Files:**
- No new files required unless verification reveals a gap

**Step 1: Run**
- `cd projects/pytorch-training-regularization-showcase && uv sync --extra dev`
- `cd projects/pytorch-training-regularization-showcase && make smoke`
- `cd projects/pytorch-training-regularization-showcase && make verify`
- `cd projects/pytorch-training-regularization-showcase && make quality`
- `cd projects/pytorch-training-regularization-showcase && make check`
- `cd projects/pytorch-training-regularization-showcase && make run`

**Step 2: Fix surfaced gaps**

**Step 3: Rerun until green**
