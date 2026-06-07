"""Student-facing bridge narrative for the adaptive learning-agent boundary."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_course_assistant_rl.agent_bridge import policy_router_payload
from adaptive_course_assistant_rl.config import DEFAULT_Q_EPISODES, QUICK_Q_EPISODES
from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    SCENARIOS,
    AssistantInterventionEnvironment,
    state_key_to_row,
)
from adaptive_course_assistant_rl.q_learning import train_q_learning

QUESTION_BY_SCENARIO: dict[int, str] = {
    0: "I keep seeing this notation in the course notes, and I don't know what it means.",
    1: "My validation score looks too good, and now I don't trust my experiment.",
    2: "I am trying to review everything before the exam, but I feel overloaded.",
    3: "I keep making study plans and then drifting away from them.",
    4: "The concept is advanced, and I kind of get it, but I still feel unsure.",
}


@dataclass(frozen=True)
class LearningAgentStory:
    """Human-readable bridge from deterministic assistant to learned controller."""

    markdown: str


def build_learning_agent_story(*, quick: bool, scenario_id: int = 1, seed: int = 19) -> LearningAgentStory:
    """Render one concrete learning-agent story for students.

    The story stays inside the adaptive showcase. It does not import the
    agentic-course-assistant runtime, but it uses the same boundary: a
    deterministic assistant understands the request first, and a learned policy
    selects the next intervention.
    """

    if scenario_id not in QUESTION_BY_SCENARIO:
        raise ValueError(f"Unsupported scenario_id: {scenario_id}")

    q_result = train_q_learning(
        episodes=QUICK_Q_EPISODES if quick else DEFAULT_Q_EPISODES,
        seed=seed,
    )
    policy = q_result.greedy_policy()
    environment = AssistantInterventionEnvironment()
    state = environment.reset(seed=seed, scenario_id=scenario_id)
    action = policy.select_action(state)
    transition = environment.step(action)
    scenario = SCENARIOS[scenario_id]
    state_row = state_key_to_row(state.as_tuple())
    router = policy_router_payload()
    action_name = ACTION_LABELS[action]

    markdown = (
        "# Learning Agent Story\n\n"
        "This artifact answers one precise question:\n\n"
        "> if you wrapped the course assistant in an agent framework such as the OpenAI Agents SDK, "
        "where would the learned policy actually fit?\n\n"
        "## The student question\n\n"
        f"{QUESTION_BY_SCENARIO[scenario_id]}\n\n"
        "## What the deterministic assistant already did\n\n"
        "- Classified the tutoring situation before the learned policy acted.\n"
        f"- Identified the scenario as `{scenario.name}`.\n"
        f"- Fixed the decision boundary to `{router['decision_boundary']}`.\n"
        "- Kept answer generation, tool wiring, and handoff logic outside the policy learner.\n\n"
        "## The state the policy sees\n\n"
        f"- intent_type: `{state_row['intent_type']}`\n"
        f"- difficulty_level: `{state_row['difficulty_level']}`\n"
        f"- confidence_level: `{state_row['confidence_level']}`\n"
        f"- misconception_type: `{state_row['misconception_type']}`\n"
        f"- retrieval_quality: `{state_row['retrieval_quality']}`\n"
        f"- intent_uncertainty: `{state_row['intent_uncertainty']}`\n"
        f"- cognitive_load: `{state_row['cognitive_load']}`\n"
        f"- attempt_count: `{state_row['attempt_count']}`\n"
        f"- safety_risk: `{state_row['safety_risk']}`\n\n"
        "## What the learned policy chose\n\n"
        f"- policy: `{policy.name}`\n"
        f"- selected action: `{action_name}`\n"
        f"- next solved flag: `{transition.state.resolved_flag}`\n"
        f"- next safety risk: `{transition.state.safety_risk}`\n"
        f"- immediate reward: `{round(transition.reward, 4)}`\n\n"
        "## How an OpenAI Agents SDK agent would use that choice\n\n"
        "- The SDK agent would still own orchestration, tools, handoffs, and trace collection.\n"
        "- The learned controller would return only the next intervention label.\n"
        f"- In this example, the SDK agent would interpret `{action_name}` and then execute the corresponding assistant behavior.\n"
        "- That means the framework agent is not itself being trained here; it is hosting a learned decision module.\n\n"
        "## What this project does and does not claim\n\n"
        "- This is a learning agent story for **intervention selection**, not full answer generation.\n"
        "- DQN and PPO exist in this project as optional DRL comparison baselines, not as the default teaching path.\n"
        "- This is **not** MARL: there is one reward-seeking policy in the simulator, not multiple learning agents with coordination or competition.\n"
        "- This is also not RLHF, DPO, or end-to-end LLM fine-tuning.\n\n"
        "## Why the simulator still matters\n\n"
        "- The scenario set is small and hand-authored on purpose.\n"
        "- The transition rules are simplified so students can inspect why the policy gets rewarded.\n"
        "- That makes the results readable, but it also means you should treat them as a teaching control sample, not deployment proof.\n\n"
        "## Where PPO fits\n\n"
        "- PPO is the optional actor-critic anchor in this repository.\n"
        "- Use it when you want to show the deep policy-gradient side of the ladder after Q-learning and DQN make sense.\n"
        "- Do not let PPO become the first thing students look at, because it hides the simple control story this artifact is trying to teach.\n\n"
        "## Governance note\n\n"
        "- If you later connect this boundary to a live SDK agent, keep offline evaluation, trace review, and human escalation in place.\n"
        "- A model-backed orchestration layer does not remove the need for policy-governance evidence.\n"
    )
    return LearningAgentStory(markdown=markdown)
