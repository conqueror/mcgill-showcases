# Method Notes

- Epsilon-greedy: simple baseline balancing random exploration.
- UCB1: optimistic confidence bounds for structured exploration.
- Thompson sampling: Bayesian posterior sampling for efficient exploration.

Arm probabilities and epsilon must be finite and between 0 and 1. Policies require at least one arm, valid arm indices, and binary rewards (0 or 1).
