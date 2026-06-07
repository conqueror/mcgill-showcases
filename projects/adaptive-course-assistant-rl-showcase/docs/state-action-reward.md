# State, Action, Reward

This is the contract the whole project rests on.

## State

The state tracks:

- intent type,
- difficulty level,
- confidence level,
- misconception type,
- retrieval quality,
- intent uncertainty,
- cognitive load,
- turn index,
- attempt count,
- last action,
- safety risk,
- resolved flag.

That looks like a lot, but each field has a job. We want enough structure to talk about pacing, grounding, and safety without turning the simulator into a monster.

## Actions

The assistant can:

- ask a clarifying question,
- retrieve a course note,
- give a hint,
- give a worked example,
- assign targeted practice,
- check understanding,
- slow down and rephrase,
- escalate to a human.

## Reward

The good reward favors:

- grounded resolution,
- lower safety risk,
- higher confidence when it is earned.

It also charges for:

- extra turns,
- intervention cost,
- ungrounded help,
- intervention churn,
- missed escalation.

That last part matters. A tutoring policy that feels helpful but keeps bluffing without evidence is exactly the kind of thing a classroom assistant should not do.

## Bandit Reward vs MDP Reward

The contextual bandit uses a separate one-step Bernoulli reward model because it is teaching one narrow question:

> given this first-turn context, which intervention should happen first?

The multi-turn algorithms use the MDP reward described above because they need to account for later states, extra turns, grounding, resolution, and escalation.

This distinction is intentional. If we used the same multi-turn transition story for the bandit, it would stop being a clean contextual-bandit example.

## Why This Is Not MARL

Only one policy is learning in this simulator.

The deterministic assistant, the artifact writer, and a future SDK runtime are not separate reward-seeking agents here. They are support infrastructure around one learned controller.

That keeps the project focused on the question students can actually inspect:

> what should the assistant do next?
