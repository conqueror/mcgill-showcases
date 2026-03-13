# Deep Learning Showcase Series Design

Date: 2026-03-13
Status: Draft for review
Scope: Design only. No implementation or scaffolding is covered by this document.

## Summary

This design proposes a three-project deep learning showcase series under `projects/` that converts the current McGill deck material into self-guided, runnable, beginner-friendly educational projects.

The recommended split is:

1. `deep-learning-math-foundations-showcase`
2. `neural-network-foundations-showcase`
3. `pytorch-training-regularization-showcase`

This is intentionally not a one-repo-per-deck translation. The original deck boundaries are useful as source material, but they are not the best learning boundaries for showcase projects. The projects should be organized by learning progression and runnable artifact design, not by slide file count.

## Problem

The source material exists as slide-style markdown decks:

- `Essential Math for Deep Learning.md`
- `Introduction to Neural Networks.md`
- `Training Feed-Forward Neural Networks.md`
- `Implementing Neural Networks in PyTorch.md`
- `Optimization Techniques & Regularization for Deep Learning.md`

Those decks contain solid raw material, but slides alone are not enough for a good showcase project because they:

- compress ideas for lecture delivery rather than self-study,
- repeat concepts across decks,
- mix conceptual, mathematical, and implementation concerns,
- do not consistently map each concept to runnable outputs,
- are not yet shaped to match the repository's existing showcase structure.

## Goals

- Create projects that match the existing `projects/` conventions in this repo.
- Translate the deck material into self-guided learning experiences.
- Keep the material educational, useful, easy to read, and easy to understand.
- Use strong documentation, clear naming, and high-quality docstrings.
- Ensure each concept is paired with runnable code and inspectable artifacts.
- Make the series suitable for beginners while still being technically honest.

## Non-Goals

- Preserve every slide verbatim.
- Cover every advanced deep learning topic in v1.
- Support multiple major frameworks equally in the main learning path.
- Turn the projects into research-grade training systems.
- Build a single monolithic "all deep learning" showcase.

## Source-to-Series Mapping

| Source deck | Primary destination | Notes |
|---|---|---|
| `Essential Math for Deep Learning.md` | `deep-learning-math-foundations-showcase` | Becomes prerequisites and intuition-building code labs. |
| `Introduction to Neural Networks.md` | `neural-network-foundations-showcase` | Becomes conceptual introduction to neural nets and their components. |
| `Training Feed-Forward Neural Networks.md` | `neural-network-foundations-showcase` and `pytorch-training-regularization-showcase` | Split conceptual training ideas from implementation-heavy labs. |
| `Implementing Neural Networks in PyTorch.md` | `pytorch-training-regularization-showcase` | PyTorch-first implementation track. Keras comparison moves to appendix or notes. |
| `Optimization Techniques & Regularization for Deep Learning.md` | `pytorch-training-regularization-showcase` | Core training and stabilization experiments live here. |

## Why Three Projects

### Why not one project

One project would become too broad and would force beginners to jump between:

- prerequisite math,
- neural network intuition,
- framework mechanics,
- optimization details,
- regularization experiments.

That would make the README, docs, artifacts, and tests harder to navigate and would weaken the self-guided structure.

### Why not five projects

Five projects would follow the deck boundaries too literally and create avoidable duplication in:

- environment setup,
- documentation patterns,
- toy dataset handling,
- training utilities,
- repeated explanations of the same core ideas.

### Why three is the right boundary

Three projects create a clear progression:

1. learn the math ideas,
2. understand neural networks conceptually,
3. implement and improve them in PyTorch.

This matches how a learner actually builds understanding and keeps each project focused enough to remain readable.

## Target Audience

Primary audience:

- beginner to early-intermediate learners,
- students moving from traditional ML into deep learning,
- self-guided learners who need runnable examples more than lecture notes.

Secondary audience:

- instructors who want a course companion,
- practitioners who want a clean refresher on fundamentals.

## Core Design Principles

Every project in the series should follow these rules:

1. One concept, one runnable anchor.
2. Prefer small, legible examples over broad but shallow coverage.
3. Explain intuition before formalism.
4. Keep theory and code tightly linked.
5. Make outputs inspectable through files in `artifacts/`.
6. Write for self-study, not for a live lecturer.
7. Prefer local, reproducible datasets and deterministic scripts where possible.
8. Use minimal but meaningful diagrams.

## Standard Learning Unit

Each major concept should be converted from "slide content" into a reusable educational unit with this structure:

1. Plain-language intuition
2. Minimal math or rule
3. Small runnable code example
4. Expected output and how to read it
5. Common mistake or misconception
6. One short exercise
7. One checkpoint question

This is the core translation pattern from deck material to showcase material.

## Repository Shape

Each project should follow the same top-level pattern already used in stronger showcases in this repo:

```text
<project>/
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── docs/
├── scripts/
├── src/<package_name>/
├── tests/
├── artifacts/
├── data/raw/.gitkeep
└── data/processed/.gitkeep
```

Optional per project:

- `notebooks/` if there is clear interactive value,
- `docs/diagrams/` if Mermaid or static diagrams meaningfully help,
- no notebook-only logic; the script path remains primary.

## Shared Documentation Set

Each project should include a consistent documentation set, adapted to its scope:

- `README.md`
- `docs/learning-flow.md`
- `docs/concept-learning-map.md`
- `docs/code-examples.md`
- `docs/domain-use-cases.md`
- `docs/checkpoint-answer-key.md`

Optional when helpful:

- `docs/diagrams.md`
- `docs/glossary.md`
- `docs/debugging-guide.md`
- `docs/student-worksheet.md`

## Documentation Standards

- Use short sections and direct language.
- Avoid slide shorthand and unexplained jargon.
- All public functions and classes should have docstrings.
- Prefer docstrings that explain purpose, inputs, outputs, and teaching role.
- Add comments only where code intent would otherwise be non-obvious.
- Keep README files task-oriented: what to learn, how to run, what to inspect next.

## Project 1: `deep-learning-math-foundations-showcase`

### Purpose

Teach the math required for deep learning without making the project feel like a generic math course.

### Learning outcomes

By the end of this project, a learner should be able to:

- explain vectors and matrices in model terms,
- understand dot products as weighted combination mechanisms,
- read derivatives as sensitivity,
- explain partial derivatives and gradients,
- understand integrals at a high level,
- connect probability and statistics to data uncertainty,
- explain entropy and cross-entropy intuitively,
- understand gradient descent before seeing full neural network training.

### Suggested module layout

```text
src/deep_learning_math_foundations_showcase/
├── __init__.py
├── config.py
├── linear_algebra.py
├── calculus.py
├── probability.py
├── information_theory.py
├── optimization.py
├── plots.py
└── reporting.py
```

### Script entrypoints

- `scripts/run_showcase.py`
- `scripts/verify_artifacts.py`

Optional:

- `scripts/validate_mermaid.sh`
- `scripts/validate_markdown_links.sh`

### Example artifacts

- `artifacts/vector_operations.csv`
- `artifacts/matrix_transformations.csv`
- `artifacts/derivative_examples.csv`
- `artifacts/gradient_descent_trace.csv`
- `artifacts/probability_simulations.csv`
- `artifacts/information_theory_summary.md`
- `artifacts/summary.md`

### Documentation emphasis

- "math for deep learning" rather than "math for math's sake",
- intuitive explanation of why each concept matters for training,
- diagrams that connect gradients, loss, and parameter updates.

### Data approach

- mostly synthetic or generated data,
- no heavy external dataset requirement.

## Project 2: `neural-network-foundations-showcase`

### Purpose

Teach what a neural network is, how its parts work together, and how feed-forward models learn before the learner has to manage a large framework-heavy codebase.

### Learning outcomes

By the end of this project, a learner should be able to:

- explain the role of neurons, layers, weights, and biases,
- compare common activation functions,
- distinguish forward pass, loss computation, and backpropagation,
- understand why initialization matters,
- explain underfitting vs overfitting,
- connect loss curves to training quality,
- evaluate a simple feed-forward network on toy tasks.

### Suggested module layout

```text
src/neural_network_foundations_showcase/
├── __init__.py
├── config.py
├── data.py
├── activations.py
├── losses.py
├── networks.py
├── backprop.py
├── training.py
├── plots.py
└── reporting.py
```

### Script entrypoints

- `scripts/run_showcase.py`
- `scripts/verify_artifacts.py`

Optional:

- `scripts/validate_markdown_links.sh`
- `scripts/validate_notebooks.sh`

### Example artifacts

- `artifacts/activation_comparison.csv`
- `artifacts/loss_function_comparison.csv`
- `artifacts/backprop_gradient_trace.csv`
- `artifacts/initialization_comparison.csv`
- `artifacts/underfit_overfit_examples.csv`
- `artifacts/training_curves.csv`
- `artifacts/summary.md`

### Recommended implementation style

- small toy datasets,
- lightweight feed-forward models,
- visuals for decision boundaries and training behavior where helpful,
- concept-first explanations that stay close to the math foundations project.

### Content notes

- keep historical material brief and supportive, not central,
- keep CNNs and RNNs as brief "where this goes next" notes rather than main implementation paths.

## Project 3: `pytorch-training-regularization-showcase`

### Purpose

Teach how to build, train, debug, and improve neural networks in PyTorch using a realistic but still beginner-appropriate workflow.

### Learning outcomes

By the end of this project, a learner should be able to:

- use tensors and autograd,
- define models with `nn.Module`,
- load data with `Dataset` and `DataLoader`,
- implement a training and validation loop,
- compare optimizers and learning-rate behavior,
- apply dropout, weight decay, early stopping, and batch normalization,
- diagnose vanishing or exploding gradients,
- read training artifacts and make model improvement decisions.

### Suggested module layout

```text
src/pytorch_training_regularization_showcase/
├── __init__.py
├── config.py
├── data.py
├── models.py
├── training.py
├── evaluation.py
├── regularization.py
├── experiments.py
├── plots.py
└── reporting.py
```

### Script entrypoints

- `scripts/run_showcase.py`
- `scripts/run_optimizer_comparison.py`
- `scripts/run_regularization_ablation.py`
- `scripts/verify_artifacts.py`

### Example artifacts

- `artifacts/baseline_metrics.json`
- `artifacts/training_history.csv`
- `artifacts/optimizer_comparison.csv`
- `artifacts/learning_rate_schedule_comparison.csv`
- `artifacts/regularization_ablation.csv`
- `artifacts/gradient_health_report.md`
- `artifacts/error_analysis.csv`
- `artifacts/summary.md`

### Recommended implementation choices

- PyTorch is the primary framework in the main path.
- Keras and TensorFlow are mentioned only in comparison notes or appendix material in v1.
- PyTorch Lightning is deferred to optional extension material.
- GAN content is deferred entirely out of the initial scope.

### Dataset guidance

Preferred:

- MNIST or FashionMNIST for the core path,
- synthetic tabular tasks for optimizer diagnostics when needed.

Avoid in v1:

- large external datasets,
- distributed training requirements,
- infrastructure-heavy setup.

## Cross-Project Narrative

The projects should be readable independently, but they should also form a series:

1. `deep-learning-math-foundations-showcase`
2. `neural-network-foundations-showcase`
3. `pytorch-training-regularization-showcase`

Each README should state:

- whether the project can stand alone,
- which earlier project is recommended first,
- which next project continues the learning path.

## Naming and Packaging

Recommended project folder names:

- `projects/deep-learning-math-foundations-showcase`
- `projects/neural-network-foundations-showcase`
- `projects/pytorch-training-regularization-showcase`

Recommended package names:

- `deep_learning_math_foundations_showcase`
- `neural_network_foundations_showcase`
- `pytorch_training_regularization_showcase`

## Implementation Decisions

### Framework choice

Decision:

- PyTorch-first for the implementation project.

Reason:

- it aligns best with the source material's practical emphasis,
- it avoids unnecessary duplication,
- it keeps the main path focused and easier to maintain.

### Notebook policy

Decision:

- notebooks are optional and secondary.

Reason:

- runnable scripts and artifacts are easier to test and maintain,
- notebooks should reinforce the learning path, not replace it.

### Dataset policy

Decision:

- prefer local, built-in, or easily downloadable small datasets.

Reason:

- this matches the repo's beginner-friendly structure,
- it reduces setup friction,
- it keeps the focus on concepts instead of data plumbing.

### Advanced topic policy

Decision:

- defer CNN/RNN/GAN/Lightning/distributed training from the core v1 path.

Reason:

- they expand scope sharply,
- they would dilute the core fundamentals,
- they fit better as future extension showcases.

## Quality Bar

Every project should meet this minimum bar:

- clear `README.md` with learning outcomes and quickstart,
- deterministic script entrypoints,
- artifact verification script,
- unit tests for core logic,
- docstrings on public APIs,
- concept-to-artifact documentation in `docs/`,
- no hidden magic in notebook cells that does not exist in `src/` or `scripts/`.

## Acceptance Criteria

The design is complete when the eventual implementation delivers:

1. Three project folders under `projects/`.
2. Each project follows the existing showcase structure.
3. Each project has a strong README and guided docs set.
4. Each major concept has a runnable example and inspectable artifact.
5. PyTorch is used only where it adds value to the learning progression.
6. Advanced topics are clearly deferred rather than half-included.
7. Tests and artifact verification exist for each project.

## Suggested Delivery Order

### Phase 1

Build `deep-learning-math-foundations-showcase` first.

Reason:

- it establishes the doc pattern,
- it has the least framework overhead,
- it creates reusable explanation assets for the later projects.

### Phase 2

Build `neural-network-foundations-showcase`.

Reason:

- it uses the math project as prerequisite context,
- it establishes the conceptual bridge to training.

### Phase 3

Build `pytorch-training-regularization-showcase`.

Reason:

- it depends most heavily on stable teaching and documentation patterns from the first two projects,
- it has the highest implementation surface area.

## Risks and Mitigations

### Risk: scope creep from source decks

Mitigation:

- organize by learning outcome, not slide order,
- explicitly defer advanced topics.

### Risk: too much theory, not enough runnable learning

Mitigation:

- require at least one artifact per major concept cluster,
- write docs around outputs, not only formulas.

### Risk: duplicated explanations across projects

Mitigation:

- define project boundaries clearly,
- link backward and forward across the series instead of re-teaching everything.

### Risk: framework comparison dilutes the main path

Mitigation:

- keep Keras/TensorFlow as appendix material in v1.

## Open Review Questions

These should be confirmed before implementation starts:

1. Is the three-project split the right pedagogical boundary for your intent?
2. Do you want Keras comparison material retained in v1, or deferred completely?
3. Do you want notebooks in the first implementation pass, or should they follow after the script-based path is solid?
4. Should the tone be strictly course-like, portfolio-like, or a hybrid of both?

## Recommended Next Step

After this spec is approved, create an implementation plan for Phase 1 only:

- scaffold `deep-learning-math-foundations-showcase`,
- define its exact files,
- map the first source deck section-by-section into docs, modules, tests, and artifacts.
