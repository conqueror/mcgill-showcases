# How The Loop Works

Autoresearch is small on purpose. The idea is not to build a giant training platform. The idea is to give an agent a narrow research surface and a clear success metric.

## The three-file mental model

### `prepare.py`

This file is the fixed harness.

It handles:

- data download,
- tokenizer setup,
- runtime utilities,
- evaluation support,
- fixed constants such as sequence length and time budget.

In the upstream workflow, the agent is not supposed to modify this file.

### `train.py`

This is the editable research surface.

It contains:

- the model,
- the optimizer,
- the training loop,
- hyperparameters,
- architecture choices.

This is where the agent experiments.

### `program.md`

This file is the human-authored instruction surface.

It defines:

- how the agent should behave,
- what files are in scope,
- how to log results,
- when to keep or discard a run,
- how autonomous the loop should be.

## Why the 5-minute budget matters

Every run gets the same wall-clock training budget. That creates a fairer comparison than "train until convergence" because the question is:

> What is the best model or training strategy this machine can find within this fixed amount of time?

That changes the optimization target. A larger model is not automatically better if it spends the whole budget compiling or moving fewer tokens.

## Why `val_bpb` is the score

The upstream repos use validation bits per byte:

- lower is better,
- it stays meaningful across tokenizer or vocabulary changes,
- it gives a single number for keep-or-discard decisions.

The loop is intentionally simple:

1. edit `train.py`,
2. run one fixed-budget experiment,
3. measure `val_bpb`,
4. keep or discard,
5. log the result.

## Why simplicity is part of the decision

The goal is not "lowest score at any cost." The upstream prompt also values simpler code.

That means:

- small gains with lots of complexity may not be worth keeping,
- equal performance with simpler code can be a win,
- crashes are useful information but should not advance the branch.

## Why platform details matter

The loop stays conceptually the same across macOS and Unix, but the implementation details change:

- macOS focuses on MPS-safe paths and SDPA-based attention fallbacks,
- Unix focuses on CUDA, FlashAttention-3, and `torch.compile`.

The research loop is stable. The systems constraints are not.
