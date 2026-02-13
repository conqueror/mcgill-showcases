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
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_arms(self) -> int:
        return len(self.arm_probs)

    @property
    def optimal_mean_reward(self) -> float:
        return float(max(self.arm_probs))

    def pull(self, arm: int) -> float:
        prob = self.arm_probs[arm]
        return float(self._rng.binomial(1, prob))
