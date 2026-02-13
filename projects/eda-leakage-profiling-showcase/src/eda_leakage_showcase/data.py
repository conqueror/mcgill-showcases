from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    target: pd.Series


def make_dataset(*, n_samples: int = 1200, random_state: int = 42) -> DatasetBundle:
    rng = np.random.default_rng(random_state)
    amount = rng.gamma(shape=2.0, scale=120.0, size=n_samples)
    tenure = rng.integers(1, 72, size=n_samples)
    segment = rng.choice(["A", "B", "C"], size=n_samples, p=[0.5, 0.3, 0.2])
    region = rng.choice(["east", "west", "north", "south"], size=n_samples)
    event_time = np.arange(n_samples)

    logits = -2.5 + 0.01 * amount + 0.02 * tenure + (segment == "C") * 0.8
    probs = 1.0 / (1.0 + np.exp(-logits))
    target = pd.Series(rng.binomial(1, probs), name="target")

    frame = pd.DataFrame(
        {
            "amount": amount,
            "tenure_months": tenure,
            "segment": segment,
            "region": region,
            "event_time": event_time,
            "group_id": rng.integers(0, 40, size=n_samples),
        }
    )
    frame.loc[frame.index[::17], "amount"] = np.nan
    frame.loc[frame.index[::23], "segment"] = None

    # Controlled leakage column for demonstration.
    frame["leak_target_copy"] = target.to_numpy()
    return DatasetBundle(frame=frame, target=target)
