# Checkpoint Answer Key

Use this only after you try the questions yourself.

## 1. Why is a fixed time budget important?

Because it makes experiments comparable under the same wall-clock constraint. The question becomes "what is best in 5 minutes on this machine?" instead of "what eventually converges with enough time?"

## 2. Why is `prepare.py` fixed?

Because it contains the data and evaluation harness. If the agent were allowed to change it, the benchmark itself could drift and results would stop being comparable.

## 3. Why might a slightly better `val_bpb` still be a discard?

Because the improvement may be too small relative to the complexity it adds. The loop is not only optimizing quality; it is also trying to avoid unnecessary complexity.

## 4. Why are macOS and Unix runs not directly comparable?

Because they can use different hardware, attention backends, compile paths, and throughput characteristics. Autoresearch is optimized for local improvement on a given machine.

## 5. What is the human actually controlling?

Primarily the instruction surface in `program.md`, plus the decision to accept or reject the broader workflow. The agent edits `train.py`, but the human defines the operating rules.

## 6. Why keep `results.tsv` untracked?

Because it is a local experiment ledger, not stable source code. It changes constantly and would create noisy commits.
