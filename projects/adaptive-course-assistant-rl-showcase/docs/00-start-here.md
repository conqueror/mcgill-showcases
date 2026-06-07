# Start Here

This project is easiest to understand if you think of it as a **bridge showcase**.

The deterministic assistant already does the front half of the work:

- figure out what kind of question this is,
- grab a reasonable starting resource,
- keep the response inside a safe teaching scope.

The RL layer decides only the next intervention.

## Map Of The Project

| Surface | What it is for |
|---|---|
| `src/adaptive_course_assistant_rl/environment.py` | The tutoring simulator: state, actions, transition, reward, and horizon |
| `src/adaptive_course_assistant_rl/contextual_bandit.py` | First-turn intervention choice only |
| `src/adaptive_course_assistant_rl/q_learning.py` | Tabular off-policy control |
| `src/adaptive_course_assistant_rl/sarsa.py` | Tabular on-policy control |
| `src/adaptive_course_assistant_rl/policy_gradient.py` | Small REINFORCE implementation |
| `src/adaptive_course_assistant_rl/drl.py` | Optional DQN and PPO bridge |
| `scripts/` | Copy-paste runnable entrypoints |
| `artifacts/` | Stable outputs the verifier checks |

## Suggested Reading Order

1. Read [system-boundary.md](system-boundary.md).
2. Read [state-action-reward.md](state-action-reward.md).
3. Run `make smoke`.
4. Open `artifacts/bandit/contextual_policy_metrics.csv`.
5. Open `artifacts/mdp/sample_episodes.csv`.
6. Open `artifacts/eval/offline_policy_eval.csv`.
7. Read [algorithm-ladder.md](algorithm-ladder.md).

## First Questions To Ask Yourself

- What part of the assistant is fixed, and what part is learned?
- Which interventions look harmless at first but become expensive or risky later?
- Does a stronger reward number actually mean a safer tutoring policy?
