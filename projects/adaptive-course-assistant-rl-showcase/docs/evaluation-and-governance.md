# Evaluation And Governance

A good-looking reward number is not a launch plan.

This project treats evaluation as a separate step because tutoring systems can fail in ways that reward alone hides.

## What We Check

`artifacts/eval/offline_policy_eval.csv` is "offline" in the local-showcase sense: the policies are replayed in a simulator without contacting students or live services. It is not formal off-policy evaluation over a logged behavior-policy dataset.

- average reward,
- solved rate,
- final safety risk,
- intervention cost,
- intervention switches,
- escalation count,
- ungrounded action count,
- whether repeated episodes produced genuinely new trajectories or only replayed the same deterministic rollout.

## Why That Matters

A policy can get paid for moving fast while still:

- overusing worked examples,
- skipping grounding,
- delaying escalation too long,
- bouncing from one intervention to another.

That is why the final artifact is not a cheerful "deployment success" note. It is a deployment recommendation with caution built in.

## Simulator Limits You Should Keep In View

- The scenario set is small and hand-authored.
- The transition rules are simplified on purpose.
- The reward is a teaching approximation, not a classroom truth metric.
- The RL family comparison is a teaching comparison on shared scenario families, not a benchmark-grade held-out study.
- When a deterministic policy sees the same scenario repeatedly, multiple episodes can collapse to the same trajectory. Treat those as replayed evidence, not independent samples.

That does not make the project weak. It makes the project readable.

It does mean you should treat the learned policy as a teaching control sample, not as proof that an SDK-hosted tutoring agent is ready for real students.
