from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class RankingDataset:
    frame: pd.DataFrame
    feature_frame: pd.DataFrame
    feature_names: list[str]
    relevance: pd.Series


def _relevance_from_points(points: pd.Series) -> NDArray[np.int64]:
    pct = points.rank(pct=True, method="average").to_numpy(dtype=np.float64)
    relevance = np.zeros_like(pct, dtype=np.int64)
    relevance[(pct >= 0.50) & (pct < 0.75)] = 1
    relevance[(pct >= 0.75) & (pct < 0.90)] = 2
    relevance[pct >= 0.90] = 3
    return relevance


def make_synthetic_player_dataset(
    *,
    n_seasons: int = 6,
    players_per_season: int = 120,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_seasons < 4:
        raise ValueError("n_seasons must be at least 4 for train/val/test grouped splitting.")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []

    for season_offset in range(n_seasons):
        season_label = f"season_{2018 + season_offset}"
        scoring_multiplier = 0.9 + (season_offset * 0.035)

        goals = rng.poisson(lam=18.0 * scoring_multiplier, size=players_per_season)
        assists = rng.poisson(lam=23.0 * scoring_multiplier, size=players_per_season)
        shots = rng.poisson(lam=165.0 * scoring_multiplier, size=players_per_season)
        toi = rng.normal(loc=15.5, scale=2.5, size=players_per_season).clip(min=6.0)
        plus_minus = rng.normal(loc=0.0, scale=9.0, size=players_per_season)

        positions = rng.choice(
            ["F", "D", "G"],
            p=[0.57, 0.35, 0.08],
            size=players_per_season,
        )
        team_tier = rng.choice(
            ["top", "mid", "bottom"],
            p=[0.25, 0.50, 0.25],
            size=players_per_season,
        )

        for idx in range(players_per_season):
            points = float(goals[idx] + assists[idx])
            rows.append(
                {
                    "season": season_label,
                    "player_id": f"{season_label}_p{idx:03d}",
                    "goals": float(goals[idx]),
                    "assists": float(assists[idx]),
                    "shots": float(shots[idx]),
                    "toi_minutes": float(toi[idx]),
                    "plus_minus": float(plus_minus[idx]),
                    "position": str(positions[idx]),
                    "team_tier": str(team_tier[idx]),
                    "points": points,
                }
            )

    frame = pd.DataFrame(rows)
    for col in ["shots", "toi_minutes", "plus_minus"]:
        missing_mask = rng.random(len(frame)) < 0.03
        frame.loc[missing_mask, col] = np.nan

    return frame


def prepare_ranking_dataset(frame: pd.DataFrame, *, group_col: str = "season") -> RankingDataset:
    if group_col not in frame.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if "points" not in frame.columns:
        raise KeyError("Expected 'points' column for relevance generation.")

    work = frame.copy()
    work[group_col] = work[group_col].astype(str)
    work["relevance"] = work.groupby(group_col, sort=False)["points"].transform(
        lambda col: pd.Series(_relevance_from_points(col), index=col.index)
    )

    numeric_cols = ["goals", "assists", "shots", "toi_minutes", "plus_minus"]
    categorical_cols = ["position", "team_tier"]

    features = work[numeric_cols + categorical_cols].copy()
    for col in numeric_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
        features[col] = features[col].fillna(float(features[col].median(skipna=True)))

    for col in categorical_cols:
        features[col] = features[col].astype(str).fillna("UNKNOWN")

    encoded = pd.get_dummies(features, columns=categorical_cols, dummy_na=False).astype(float)

    return RankingDataset(
        frame=work,
        feature_frame=encoded,
        feature_names=list(encoded.columns),
        relevance=work["relevance"].astype(float),
    )
