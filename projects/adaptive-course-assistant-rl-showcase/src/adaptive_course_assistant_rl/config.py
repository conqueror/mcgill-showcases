"""Project-level defaults for the adaptive course assistant RL showcase.

The rest of the package imports these constants instead of hard-coding them so the
smoke path, the full run, and the optional deep-RL bridge stay in sync.
"""

from __future__ import annotations

DEFAULT_HORIZON = 5
DEFAULT_SCENARIO_IDS = (0, 1, 2, 3, 4)
DEFAULT_Q_EPISODES = 450
DEFAULT_REINFORCE_EPISODES = 700
DEFAULT_EVAL_EPISODES = 8

QUICK_Q_EPISODES = 120
QUICK_REINFORCE_EPISODES = 180
QUICK_EVAL_EPISODES = 3
QUICK_DRL_TIMESTEPS = 800
FULL_DRL_TIMESTEPS = 2800
