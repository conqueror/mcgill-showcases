from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class BanditPolicy(Protocol):
    name: str

    def select_arm(self) -> int: ...

    def update(self, arm: int, reward: float) -> None: ...


@dataclass
class EpsilonGreedy:
    n_arms: int
    epsilon: float
    seed: int = 42
    name: str = "epsilon_greedy"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.values = np.zeros(self.n_arms, dtype=float)
        self.counts = np.zeros(self.n_arms, dtype=float)

    def select_arm(self) -> int:
        if float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(0, self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


@dataclass
class UCB1:
    n_arms: int
    name: str = "ucb1"

    def __post_init__(self) -> None:
        self.values = np.zeros(self.n_arms, dtype=float)
        self.counts = np.zeros(self.n_arms, dtype=float)
        self.t = 0

    def select_arm(self) -> int:
        self.t += 1
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        bonuses = np.sqrt((2.0 * np.log(self.t)) / self.counts)
        return int(np.argmax(self.values + bonuses))

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


@dataclass
class ThompsonSampling:
    n_arms: int
    seed: int = 42
    name: str = "thompson_sampling"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.alpha = np.ones(self.n_arms, dtype=float)
        self.beta = np.ones(self.n_arms, dtype=float)

    def select_arm(self) -> int:
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        if reward >= 0.5:
            self.alpha[arm] += 1.0
        else:
            self.beta[arm] += 1.0
