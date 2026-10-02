from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BernoulliBanditEnvironment:
    arm_probs: list[float]
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.arm_probs:
            raise ValueError("arm_probs must not be empty")
        if any(not 0.0 <= probability <= 1.0 for probability in self.arm_probs):
            raise ValueError("arm probabilities must be finite and between 0 and 1")
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_arms(self) -> int:
        return len(self.arm_probs)

    @property
    def optimal_mean_reward(self) -> float:
        return float(max(self.arm_probs))

    def pull(self, arm: int) -> float:
        if not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.n_arms:
            raise ValueError("arm must be an index between 0 and n_arms - 1")
        prob = self.arm_probs[arm]
        return float(self._rng.binomial(1, prob))
