# Learning Guide

This project works best if you treat it like a lab, not just a code dump.

## Phase 1: See The Boundary

Read [system-boundary.md](system-boundary.md) first. If that boundary feels fuzzy, the rest of the project will feel fuzzy too.

## Phase 2: Understand The Simulator

Read [state-action-reward.md](state-action-reward.md), then open `artifacts/concepts/state_action_reward_schema.csv`.

Do not rush past this part. Most RL confusion comes from a weak problem definition, not from the optimizer.

## Phase 3: Start Small

Look at the contextual bandit outputs before the multi-turn learners.

That gives you a clean answer to a narrow question:

> given only the opening tutoring context, which first move looks best?

## Phase 4: Add Delayed Consequences

Once you care about turn order, later recovery, and escalation timing, the problem stops being a bandit and becomes an MDP.

That is when Q-learning, SARSA, and REINFORCE become worth reading.

## Phase 5: Treat DQN And PPO As A Bridge

The optional DRL path is useful, but it is not the heart of the project.

If the tabular path still feels murky, DQN and PPO will only make the same ideas harder to inspect.

## Phase 6: Read The Learning-Agent Story Carefully

Open `artifacts/bridge/learning_agent_story.md`.

That artifact is where the project answers a question students often ask too early:

> if I wrap this assistant in the OpenAI Agents SDK, does that automatically make it a learning agent?

The answer is no. The framework can host the policy, but the framework is not the thing being trained in this showcase.
