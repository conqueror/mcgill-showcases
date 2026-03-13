# Neural Network Foundations Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use core-executing-plans to implement this plan task-by-task.

**Goal:** Build `projects/neural-network-foundations-showcase` as a self-guided, beginner-friendly project that teaches perceptrons, layers, activations, losses, backpropagation intuition, initialization, and underfitting vs overfitting before any heavy framework engineering.

**Architecture:** The project will be a standalone showcase with a script-first learning path, NumPy-based neural network components, toy datasets, stable artifacts, and student-facing docs. The core path will use small binary classification tasks so the learner can connect forward pass, loss, and gradient updates to visible behavior and decision-boundary outputs.

**Reasoning:** This project sits between math prerequisites and framework-centric training. It should keep the code minimal enough that a learner can still understand what each tensor-like array and parameter matrix is doing without hiding the mechanics behind PyTorch abstractions.

**Tech Stack:** Python 3.11, uv, NumPy, pandas, Matplotlib, scikit-learn, pytest, Ruff, ty, Markdown

---

### Task 1: Scaffold the project structure and package path

**Files:**
- Create: `projects/neural-network-foundations-showcase/README.md`
- Create: `projects/neural-network-foundations-showcase/Makefile`
- Create: `projects/neural-network-foundations-showcase/pyproject.toml`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/__init__.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/config.py`
- Create: `projects/neural-network-foundations-showcase/scripts/__init__.py`
- Create: `projects/neural-network-foundations-showcase/scripts/run_showcase.py`
- Create: `projects/neural-network-foundations-showcase/scripts/verify_artifacts.py`
- Create: `projects/neural-network-foundations-showcase/scripts/run_quality_checks.sh`
- Create: `projects/neural-network-foundations-showcase/tests/test_package_imports.py`
- Create: `projects/neural-network-foundations-showcase/artifacts/manifest.json`
- Create: `projects/neural-network-foundations-showcase/artifacts/.gitkeep`
- Create: `projects/neural-network-foundations-showcase/data/raw/.gitkeep`
- Create: `projects/neural-network-foundations-showcase/data/processed/.gitkeep`

**Step 1: Write the failing import test**

**Step 2: Run the test to confirm the missing project**
- Command: `cd projects/neural-network-foundations-showcase && uv run pytest tests/test_package_imports.py -q`

**Step 3: Create the minimal scaffold**

**Step 4: Rerun the import test**

### Task 2: Implement toy datasets and activation/loss utilities

**Files:**
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/data.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/activations.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/losses.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_data.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_activations.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_losses.py`

**Step 1: Write failing tests for toy datasets, activations, and losses**

**Step 2: Run targeted failing tests**

**Step 3: Implement deterministic helpers**

**Step 4: Rerun targeted tests**

### Task 3: Implement network forward pass, initialization, and backprop helpers

**Files:**
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/networks.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/backprop.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_networks.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_backprop.py`

**Step 1: Write failing tests for perceptron/MLP forward pass, init strategies, and gradient traces**

**Step 2: Run targeted failing tests**

**Step 3: Implement minimal NumPy network mechanics**

**Step 4: Rerun targeted tests**

### Task 4: Implement the training loop, reporting, and decision-boundary outputs

**Files:**
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/training.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/plots.py`
- Create: `projects/neural-network-foundations-showcase/src/neural_network_foundations_showcase/reporting.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_training.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_reporting.py`

**Step 1: Write failing tests for training curves, underfit/overfit summaries, and artifact schemas**

**Step 2: Run targeted failing tests**

**Step 3: Implement the minimal training path and reporting helpers**

**Step 4: Rerun targeted tests**

### Task 5: Wire the showcase pipeline and artifact verification

**Files:**
- Modify: `projects/neural-network-foundations-showcase/scripts/run_showcase.py`
- Modify: `projects/neural-network-foundations-showcase/scripts/verify_artifacts.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_pipeline.py`
- Create: `projects/neural-network-foundations-showcase/tests/test_verify_artifacts.py`

**Step 1: Write failing pipeline and verifier tests**

**Step 2: Run targeted failing tests**

**Step 3: Implement artifact generation for:
  - `activation_comparison.csv`
  - `loss_function_comparison.csv`
  - `backprop_gradient_trace.csv`
  - `initialization_comparison.csv`
  - `underfit_overfit_examples.csv`
  - `training_curves.csv`
  - `decision_boundary_summary.csv`
  - `decision_boundaries.png`
  - `summary.md`
**

**Step 4: Rerun targeted tests**

### Task 6: Write the self-guided docs set

**Files:**
- Modify: `projects/neural-network-foundations-showcase/README.md`
- Create: `projects/neural-network-foundations-showcase/docs/learning-flow.md`
- Create: `projects/neural-network-foundations-showcase/docs/concept-learning-map.md`
- Create: `projects/neural-network-foundations-showcase/docs/code-examples.md`
- Create: `projects/neural-network-foundations-showcase/docs/domain-use-cases.md`
- Create: `projects/neural-network-foundations-showcase/docs/checkpoint-answer-key.md`
- Create: `projects/neural-network-foundations-showcase/tests/test_docs_manifest.py`

**Step 1: Write the failing docs manifest test**

**Step 2: Run the failing docs test**

**Step 3: Write the docs**

**Step 4: Rerun the docs test**

### Task 7: Run the full project verification loop

**Files:**
- No new files required unless verification reveals a gap

**Step 1: Run**
- `cd projects/neural-network-foundations-showcase && uv sync --extra dev`
- `cd projects/neural-network-foundations-showcase && make run`
- `cd projects/neural-network-foundations-showcase && make verify`
- `cd projects/neural-network-foundations-showcase && make quality`
- `cd projects/neural-network-foundations-showcase && make check`

**Step 2: Fix surfaced gaps**

**Step 3: Rerun until green**
