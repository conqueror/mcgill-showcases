from __future__ import annotations

import numpy as np
import pandas as pd


def generate_events(
    *,
    n_events: int = 1200,
    max_lateness: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic event-time data with arrival-time jitter."""
    rng = np.random.default_rng(seed)
    event_time = np.arange(n_events)
    base_value = rng.gamma(shape=2.5, scale=15.0, size=n_events)
    lateness = rng.integers(0, max_lateness + 1, size=n_events)
    direction = rng.choice([-1, 1], size=n_events, p=[0.2, 0.8])
    arrival_time = np.maximum(0, event_time + (lateness * direction))

    frame = pd.DataFrame(
        {
            "event_id": np.arange(n_events),
            "event_time": event_time,
            "arrival_time": arrival_time,
            "value": base_value.round(3),
        }
    )
    return frame.sort_values(by="arrival_time").reset_index(drop=True)
