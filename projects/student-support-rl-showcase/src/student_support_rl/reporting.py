"""Define and enforce the showcase artifact contract, plus the deploy/shadow/reject rule.

This module is the *governance and evidence* layer of the showcase: it does no learning
itself, but it pins down exactly which files every training/evaluation run must emit, writes
them to disk, validates their shape, and turns an offline-evaluation summary into a rollout
decision. Conceptually this sits at the far end of the RL ladder (contextual bandit -> MDP ->
Q-learning -> DQN -> policy gradient -> actor-critic -> PPO): once an agent has been learned,
RL practice demands a disciplined offline check *before* any deployment, and this module is
where that discipline is encoded as code rather than prose.

The artifact contract is the source of truth shared by three parties: the runner scripts that
*produce* artifacts, ``scripts/verify_artifacts.py`` that *checks* them in CI, and
``tests/test_artifact_contract.py`` that *tests* the checker. ``REQUIRED_ARTIFACTS`` and
``OPTIONAL_DRL_ARTIFACTS`` are that contract's data; do not edit them casually because the
checked-in ``artifacts/manifest.json`` and the test suite assert byte-for-byte agreement.

RL concept:
    Evaluation and governance -- offline evaluation gates and reproducible evidence.
    See docs/evaluation-and-governance.md. The recommendation rule connects to reward design
    and reward hacking (docs/reward-design-and-hacking.md): a policy that wins reward by
    over-intervening must be caught here.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# Artifact contract (data, not behavior): the exact set of relative paths a complete showcase
# run must emit. Mirrored verbatim in artifacts/manifest.json and asserted by the test suite.
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "artifacts/concepts/mdp_spec.md",
    "artifacts/concepts/concept_map.csv",
    "artifacts/concepts/algorithm_progression.md",
    "artifacts/bandit/reward_trace.csv",
    "artifacts/bandit/regret_trace.csv",
    "artifacts/mdp/sample_episodes.csv",
    "artifacts/q_learning/training_curve.csv",
    "artifacts/q_learning/q_table.csv",
    "artifacts/dp/optimal_action_values.csv",
    "artifacts/dp/q_learning_gap.csv",
    "artifacts/sarsa/training_curve.csv",
    "artifacts/sarsa/q_table.csv",
    "artifacts/policy_gradient/training_curve.csv",
    "artifacts/eval/policy_comparison.csv",
    "artifacts/eval/scenario_results.csv",
    "artifacts/reward/reward_hacking_report.md",
    "artifacts/reward/reward_spec_good.md",
    "artifacts/reward/reward_spec_bad.md",
    "artifacts/governance/safety_controls.md",
    "artifacts/governance/offline_eval_plan.md",
    "artifacts/business/deploy_shadow_reject_memo.md",
)

OPTIONAL_DRL_ARTIFACTS: tuple[str, ...] = (
    # Optional deep-RL bridge (DQN/PPO). Either none of these exist, or the whole comparison
    # group must exist together (see artifact_validation_errors for the all-or-nothing rule).
    "artifacts/drl_optional/bridge_report.md",
    "artifacts/drl_optional/rl_family_comparison.csv",
    "artifacts/drl_optional/scenario_rollups.csv",
    "artifacts/drl_optional/training_summary.csv",
    "artifacts/drl_optional/policy_gradient_notes.md",
)


def write_manifest(
    path: Path,
    required_files: Sequence[str] = REQUIRED_ARTIFACTS,
    optional_files: Sequence[str] = OPTIONAL_DRL_ARTIFACTS,
) -> None:
    """Serialize the artifact contract to a versioned JSON manifest.

    Writes the required/optional artifact lists to ``path`` so that downstream tooling can
    validate a run against a *frozen* contract instead of whatever code happens to be loaded.
    The checked-in ``artifacts/manifest.json`` is produced this way and is asserted to match
    the module-level tuples by ``tests/test_artifact_contract.py``.

    Args:
        path: Destination of the manifest JSON file (parents are created as needed).
        required_files: Relative paths that must always be present. Defaults to
            ``REQUIRED_ARTIFACTS``.
        optional_files: Relative paths for the optional deep-RL comparison group. Defaults to
            ``OPTIONAL_DRL_ARTIFACTS``.

    RL concept:
        Evaluation and governance -- a reproducible, version-pinned evidence contract.
        See docs/evaluation-and-governance.md.
    """
    payload = {
        "version": 1,
        "required_files": list(required_files),
        "optional_files": list(optional_files),
    }
    write_json_artifact(path, payload)


def write_csv_artifact(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write a list of uniform mappings as a CSV artifact.

    The header is taken from the keys of the first row, so every mapping is expected to share
    the same schema. This is the workhorse for tabular evidence (training curves, Q-tables,
    policy comparisons) whose required columns are enforced by ``_validate_csv_artifact``.

    Args:
        path: Destination CSV path (parent directories are created as needed).
        rows: Sequence of mappings sharing a common key set. An empty sequence writes an empty
            file (no header), which the validator later treats as a contract violation.

    RL concept:
        Evaluation and governance -- machine-checkable, columnar evidence.
        See docs/evaluation-and-governance.md.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json_artifact(path: Path, payload: Mapping[str, object]) -> None:
    """Write a mapping as a stably-ordered, indented JSON artifact.

    Keys are sorted and the output is indented so that diffs stay small and deterministic
    across runs -- important for the manifest, which is committed to version control and
    compared against the in-code contract in tests.

    Args:
        path: Destination JSON path (parent directories are created as needed).
        payload: Mapping to serialize; converted to a plain ``dict`` before dumping.

    RL concept:
        Evaluation and governance -- deterministic, reviewable run metadata.
        See docs/evaluation-and-governance.md.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text_artifact(path: Path, content: str) -> None:
    """Write a text/markdown artifact with normalized leading/trailing whitespace.

    Strips surrounding whitespace and guarantees exactly one trailing newline so that the
    narrative artifacts (MDP spec, governance memos, reward-hacking report) are byte-stable.
    The validators only require a leading ``#`` heading and certain key terms, both preserved
    by this normalization.

    Args:
        path: Destination text path (parent directories are created as needed).
        content: Raw text body; outer whitespace is stripped and one newline appended.

    RL concept:
        Evaluation and governance -- reproducible narrative evidence.
        See docs/evaluation-and-governance.md.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def missing_required_artifacts(
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Return the required artifact paths that do not exist under ``output_dir``.

    This is the first gate ``scripts/verify_artifacts.py`` applies: presence before shape. A
    run is incomplete if any required relative path is missing on disk.

    Args:
        output_dir: Project root that should contain the ``artifacts/`` tree.
        manifest_path: Optional frozen contract; when given and present, its required list is
            used instead of ``REQUIRED_ARTIFACTS``.

    Returns:
        The subset of required relative paths absent from ``output_dir`` (empty if complete).

    RL concept:
        Evaluation and governance -- completeness check before validation.
        See docs/evaluation-and-governance.md.
    """
    required_files = required_artifacts(manifest_path)
    return [
        relative_path
        for relative_path in required_files
        if not (output_dir / relative_path).exists()
    ]


def required_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the required-artifact list from a manifest, or fall back to the built-in tuple.

    Lets the verifier validate against either a version-pinned manifest (reproducible audits)
    or the live in-code contract (developer convenience) using the same code path.

    Args:
        manifest_path: Optional manifest JSON. Used only if it both is not ``None`` and exists
            on disk; otherwise ``REQUIRED_ARTIFACTS`` is returned.

    Returns:
        The list of required relative artifact paths.

    RL concept:
        Evaluation and governance -- a single source of truth for the evidence contract.
        See docs/evaluation-and-governance.md.
    """
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload["required_files"]]
    return list(REQUIRED_ARTIFACTS)


def optional_artifacts(manifest_path: Path | None = None) -> list[str]:
    """Resolve the optional deep-RL artifact list from a manifest, or use the built-in tuple.

    Mirror of :func:`required_artifacts` for the optional DQN/PPO comparison group. Reads the
    ``optional_files`` key defensively (defaults to empty) so older manifests without that key
    still load.

    Args:
        manifest_path: Optional manifest JSON. Used only if it both is not ``None`` and exists
            on disk; otherwise ``OPTIONAL_DRL_ARTIFACTS`` is returned.

    Returns:
        The list of optional relative artifact paths.

    RL concept:
        Evaluation and governance -- gating the optional deep-RL bridge.
        See docs/evaluation-and-governance.md and docs/deep-rl.md.
    """
    if manifest_path is not None and manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [str(item) for item in payload.get("optional_files", [])]
    return list(OPTIONAL_DRL_ARTIFACTS)


def artifact_validation_errors(
    *,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> list[str]:
    """Validate the *shape* of every present artifact and return human-readable errors.

    Runs after the presence check: each existing required artifact is validated for its
    contract (required CSV columns / required markdown headings and terms). The optional
    deep-RL group is all-or-nothing -- if *any* comparison output beyond the bridge report is
    present, then the entire ``OPTIONAL_DRL_ARTIFACTS`` group is required and validated;
    otherwise a lone ``bridge_report.md`` is validated on its own. Missing *required* files are
    skipped here because :func:`missing_required_artifacts` already reports them.

    Args:
        output_dir: Project root containing the ``artifacts/`` tree.
        manifest_path: Optional frozen contract used to resolve the required/optional lists.

    Returns:
        A list of validation error messages; empty means every present artifact is well-formed.

    RL concept:
        Evaluation and governance -- structural validation of the evidence contract.
        See docs/evaluation-and-governance.md.
    """
    errors: list[str] = []
    required_files = required_artifacts(manifest_path)

    for relative_path in required_files:
        artifact_path = output_dir / relative_path
        if not artifact_path.exists():
            continue
        errors.extend(_validate_required_artifact(relative_path, artifact_path))

    optional_files = optional_artifacts(manifest_path)
    optional_paths = {relative_path: output_dir / relative_path for relative_path in optional_files}
    present_optional = {
        relative_path: artifact_path
        for relative_path, artifact_path in optional_paths.items()
        if artifact_path.exists()
    }
    # All-or-nothing rule: any optional file *other than* the standalone bridge report signals
    # an intent to ship the full DQN/PPO comparison, which then requires the whole group.
    has_drl_comparison = any(
        relative_path != "artifacts/drl_optional/bridge_report.md"
        for relative_path in present_optional
    )
    if has_drl_comparison:
        for relative_path, artifact_path in optional_paths.items():
            if not artifact_path.exists():
                errors.append(
                    "Optional DRL comparison output is incomplete: "
                    f"missing {relative_path}."
                )
                continue
            errors.extend(_validate_optional_drl_artifact(relative_path, artifact_path))
    elif "artifacts/drl_optional/bridge_report.md" in present_optional:
        errors.extend(
            _validate_optional_drl_artifact(
                "artifacts/drl_optional/bridge_report.md",
                present_optional["artifacts/drl_optional/bridge_report.md"],
            )
        )
    return errors


def mdp_spec_markdown() -> str:
    """Render the human-readable specification of the student-support MDP.

    Produces the ``artifacts/concepts/mdp_spec.md`` body: the (state, action, transition,
    horizon, reward) tuple that defines the finite-horizon MDP every learner in the showcase
    operates on, plus the bridge narrative from tabular Bellman control up to DQN and PPO. This
    is the "bottom of the ladder made explicit" -- naming the MDP that bandit warm-up,
    Q-learning, SARSA, REINFORCE, and the optional deep-RL methods all share.

    Returns:
        Markdown text beginning with a top-level heading (satisfies the markdown validator).

    RL concept:
        MDP framing -- states, actions, transitions, finite horizon, reward.
        See docs/mdp-and-environment.md.

    Math:
        The reward after acting at week t is R_{t+1}; the discounted return is
        G_t = sum_k gamma^k R_{t+k+1}, and control methods target Q*(s,a) =
        E[R_{t+1} + gamma * max_a' Q*(s',a')].
    """
    return (
        "# Student Support Intervention MDP\n\n"
        "## Core MDP Elements\n\n"
        "- State: week, engagement, completion, pressure, risk, prior interventions.\n"
        "- Actions: do nothing, send a resource email, recommend a TA session, "
        "or escalate to an advisor meeting.\n"
        "- Transition: a small deterministic teaching simulator with diminishing returns "
        "from repeated interventions and pressure-sensitive drift.\n"
        "- Horizon: 6 weekly decisions.\n"
        "- Policy set: random, heuristic, advisor-heavy, tabular Q-learning, SARSA, "
        "tabular REINFORCE, optional DQN, and optional PPO.\n"
        "- Reward: compare a long-term aligned reward against a short-term proxy reward.\n\n"
        "## Bellman And Evaluation Bridge\n\n"
        "- Q-learning updates action values with the Bellman target.\n"
        "- SARSA learns on-policy, bootstrapping from the action actually taken next.\n"
        "- Dynamic programming (backward induction) computes the exact optimal Q* for this "
        "finite-horizon MDP as a ground-truth baseline for Q-learning.\n"
        "- DQN reuses the same control framing with a neural value approximator.\n"
        "- PPO adds an actor-critic, policy-gradient baseline on the same environment family.\n"
        "- Offline evaluation checks reward, final risk, intervention cost, escalation, "
        "and questionable actions before any rollout recommendation.\n"
    )


def algorithm_progression_markdown() -> str:
    """Render the RL-to-DRL ladder narrative for the showcase.

    Produces ``artifacts/concepts/algorithm_progression.md``: the ordered story from
    contextual bandit -> Q-learning -> dynamic programming -> SARSA -> DQN -> policy gradients
    -> actor-critic -> PPO, with pointers to the artifact that lets a reader inspect each rung.
    The validator requires the
    terms "q-learning", "dqn", "policy gradients", "actor-critic", and "ppo" to appear, so the
    rungs are named explicitly in the prose.

    Returns:
        Markdown text beginning with a top-level heading and naming each ladder rung.

    RL concept:
        The algorithm ladder -- value-based to policy-based to actor-critic.
        See docs/value-based-learning.md, docs/policy-gradient-and-actor-critic.md, and
        docs/deep-rl.md.
    """
    return (
        "# RL To DRL Progression\n\n"
        "## The ladder in this showcase\n\n"
        "1. **Contextual bandit**: make one decision from student-state features, learn "
        "which action works best for that context, and measure regret.\n"
        "2. **Tabular Q-learning**: move from one-step decisions to a multi-week MDP and "
        "learn action values with the Bellman update.\n"
        "3. **Dynamic programming**: solve the known finite-horizon MDP exactly by backward "
        "induction to get the optimal Q*, the ground truth Q-learning is compared against.\n"
        "4. **SARSA**: on-policy TD control that bootstraps from the action actually taken "
        "next instead of the greedy max.\n"
        "5. **DQN**: keep the value-learning idea from Q-learning, but replace the Q-table "
        "with a neural network so the agent can work from continuous observations.\n"
        "6. **Policy gradients**: optimize a policy directly instead of learning values first.\n"
        "7. **Actor-critic**: combine a policy learner (actor) with a value estimator "
        "(critic) to reduce variance and stabilize training.\n"
        "8. **PPO**: use an actor-critic policy-gradient method with clipped updates so "
        "the policy improves more steadily.\n\n"
        "## How to inspect each step\n\n"
        "- Contextual bandit: `artifacts/bandit/reward_trace.csv` and "
        "`artifacts/bandit/regret_trace.csv`\n"
        "- Q-learning: `artifacts/q_learning/training_curve.csv` and "
        "`artifacts/q_learning/q_table.csv`\n"
        "- Dynamic programming ground truth: `artifacts/dp/optimal_action_values.csv` and "
        "`artifacts/dp/q_learning_gap.csv`\n"
        "- SARSA (on-policy): `artifacts/sarsa/training_curve.csv`\n"
        "- Policy gradient (REINFORCE): `artifacts/policy_gradient/training_curve.csv`\n"
        "- DQN vs PPO vs Q-learning: `artifacts/drl_optional/rl_family_comparison.csv`, "
        "`artifacts/drl_optional/scenario_rollups.csv`, and "
        "`artifacts/drl_optional/training_summary.csv`\n"
        "- Policy-gradient and actor-critic notes: "
        "`artifacts/drl_optional/policy_gradient_notes.md`\n"
    )


def concept_map_rows() -> list[dict[str, str]]:
    """Build the concept-to-component-to-artifact crosswalk rows.

    Each row links one RL concept (e.g. exploration/exploitation, the Bellman update,
    on-policy vs off-policy, policy gradients, governance) to the showcase component that
    implements it and the artifact that evidences it. Serialized to
    ``artifacts/concepts/concept_map.csv``, whose required columns are ``concept``,
    ``showcase_component``, and ``artifact`` -- exactly the keys produced here. This is the
    table-of-contents for the whole ladder.

    Returns:
        A list of dicts, each with ``concept``, ``showcase_component``, and ``artifact`` keys.

    RL concept:
        Curriculum map across the ladder -- bandit to MDP to value/policy methods to deep RL.
        See docs/glossary.md and docs/evaluation-and-governance.md.
    """
    return [
        {
            "concept": "agent_environment_loop",
            "showcase_component": "StudentSupportEnvironment.reset/step/observe/is_done",
            "artifact": "artifacts/mdp/sample_episodes.csv",
        },
        {
            "concept": "mdp_framing",
            "showcase_component": "mdp_spec_markdown",
            "artifact": "artifacts/concepts/mdp_spec.md",
        },
        {
            "concept": "exploration_vs_exploitation",
            "showcase_component": "contextual epsilon-greedy bandit warm-up",
            "artifact": "artifacts/bandit/regret_trace.csv",
        },
        {
            "concept": "bellman_and_q_learning",
            "showcase_component": "train_q_learning",
            "artifact": "artifacts/q_learning/training_curve.csv",
        },
        {
            "concept": "dynamic_programming_ground_truth",
            "showcase_component": "optimal_action_values (backward induction)",
            "artifact": "artifacts/dp/q_learning_gap.csv",
        },
        {
            "concept": "on_policy_vs_off_policy",
            "showcase_component": "train_sarsa versus train_q_learning",
            "artifact": "artifacts/sarsa/training_curve.csv",
        },
        {
            "concept": "policy_gradient_reinforce",
            "showcase_component": "train_reinforce (tabular softmax)",
            "artifact": "artifacts/policy_gradient/training_curve.csv",
        },
        {
            "concept": "deep_rl_comparison",
            "showcase_component": "run_drl_comparison",
            "artifact": "artifacts/drl_optional/rl_family_comparison.csv",
        },
        {
            "concept": "policy_gradient_actor_critic",
            "showcase_component": "PPO policy-gradient notes",
            "artifact": "artifacts/drl_optional/policy_gradient_notes.md",
        },
        {
            "concept": "reward_design",
            "showcase_component": "good versus bad reward models",
            "artifact": "artifacts/reward/reward_hacking_report.md",
        },
        {
            "concept": "offline_evaluation",
            "showcase_component": "fixed scenario policy comparison",
            "artifact": "artifacts/eval/policy_comparison.csv",
        },
        {
            "concept": "governance",
            "showcase_component": "deploy-shadow-reject framing",
            "artifact": "artifacts/business/deploy_shadow_reject_memo.md",
        },
    ]


def governance_artifacts() -> dict[str, str]:
    """Build the governance narrative artifacts: safety controls, eval plan, and rollout memo.

    Returns the markdown bodies written to ``artifacts/governance/safety_controls.md``,
    ``artifacts/governance/offline_eval_plan.md``, and
    ``artifacts/business/deploy_shadow_reject_memo.md``. These encode the human-oversight and
    offline-gating discipline that wraps any RL deployment, and they articulate the same
    over-intervention concern that :func:`recommendation_from_summary` enforces numerically.

    Returns:
        A mapping with keys ``safety_controls``, ``offline_eval_plan``, and ``business_memo``,
        each a markdown string beginning with a top-level heading.

    RL concept:
        Evaluation and governance -- safety controls and the deploy/shadow/reject framing.
        See docs/evaluation-and-governance.md and docs/reward-design-and-hacking.md.
    """
    return {
        "safety_controls": (
            "# Safety Controls\n\n"
            "- Use synthetic students only.\n"
            "- Keep advisor escalation human-reviewed in any real deployment.\n"
            "- Run offline evaluation before shadow or live rollout.\n"
        ),
        "offline_eval_plan": (
            "# Offline Evaluation Plan\n\n"
            "1. Hold scenarios fixed across policies.\n"
            "2. Compare reward, final risk, and intervention volume.\n"
            "3. Reject any policy that improves reward only by over-intervening.\n"
        ),
        "business_memo": (
            "# Deploy, Shadow, or Reject Memo\n\n"
            "Recommendation: shadow first.\n\n"
            "Reason: the tabular agent outperforms the random baseline offline, but the "
            "reward-design comparison shows clear reward-hacking risk if governance is weak.\n"
        ),
    }


def recommendation_from_summary(
    summary_rows: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    """Decide deploy/shadow/reject for the learned policy from offline-evaluation summaries.

    Implements the showcase's rollout gate -- the practical payoff of putting evaluation and
    governance at the top of the ladder. Intuition: a learned policy earns a guarded rollout
    only if it is both *safe enough* and *better than the incumbent heuristic*; anything else
    is rejected. The rule, applied to the ``q_learning`` row using ``heuristic`` as baseline:

    1. Reject if it is unsafe -- ``unsafe_rate > 0.5`` or ``final_risk > 1.6``.
    2. Otherwise shadow if it beats the heuristic -- ``avg_reward > baseline_reward`` while
       staying within the safety bound (``unsafe_rate <= 0.5``).
    3. Otherwise reject for an insufficient margin over the heuristic.

    The unsafe-rate gate is what prevents accepting a reward-hacking policy that inflates
    reward by over-intervening (the failure mode the reward-design comparison demonstrates).

    Args:
        summary_rows: Per-policy offline-eval summary mappings. Must include rows whose
            ``policy`` is ``"q_learning"`` and ``"heuristic"``, each carrying ``avg_reward``,
            ``avg_unsafe_or_questionable_decisions``, and ``avg_final_risk`` fields.

    Returns:
        A ``(decision, rationale)`` tuple where ``decision`` is one of ``"shadow"`` or
        ``"reject"``. Returns ``("reject", ...)`` if either required policy row is absent.

    RL concept:
        Evaluation and governance -- the deploy/shadow/reject decision rule, with a guard
        against reward hacking. See docs/evaluation-and-governance.md and
        docs/reward-design-and-hacking.md.
    """
    by_policy = {str(row["policy"]): row for row in summary_rows}
    q_learning = by_policy.get("q_learning")
    heuristic = by_policy.get("heuristic")
    if q_learning is None or heuristic is None:
        return ("reject", "Missing comparable heuristic or learned policy evidence.")

    avg_reward = _coerce_float(q_learning["avg_reward"])
    baseline_reward = _coerce_float(heuristic["avg_reward"])
    unsafe_rate = _coerce_float(q_learning["avg_unsafe_or_questionable_decisions"])
    final_risk = _coerce_float(q_learning["avg_final_risk"])

    # Safety gate: reject reward-hacking / high-residual-risk policies before comparing reward.
    if unsafe_rate > 0.5 or final_risk > 1.6:
        return (
            "reject",
            "The learned policy still shows too much safety or residual-risk exposure.",
        )
    # Shadow only if the learned policy beats the heuristic baseline within the safety bound.
    if avg_reward > baseline_reward and unsafe_rate <= 0.5:
        return (
            "shadow",
            "The learned policy beats the heuristic offline, but still needs guarded rollout.",
        )
    return (
        "reject",
        "The learned policy does not clearly outperform the heuristic with enough margin.",
    )


def _coerce_float(value: object) -> float:
    """Coerce a numeric-or-string artifact value to ``float``, rejecting other types.

    Summary rows may arrive with values typed as ``int``/``float`` (in-memory) or ``str``
    (round-tripped through CSV), so both are accepted; anything else is a programming error
    and raises rather than silently mis-parsing.

    Args:
        value: The value to coerce.

    Returns:
        The value as a ``float``.

    Raises:
        TypeError: If ``value`` is not an ``int``, ``float``, or ``str``.
    """
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected int, float, or str value, got {type(value)!r}")


def _validate_required_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Dispatch a required artifact to the CSV or markdown validator by file extension.

    Args:
        relative_path: The artifact's contract-relative path (also selects its rules).
        artifact_path: Absolute/resolved path to the file on disk.

    Returns:
        Validation error messages for this artifact (empty if it passes or is neither
        ``.csv`` nor ``.md``).

    RL concept:
        Evaluation and governance -- per-artifact contract enforcement.
        See docs/evaluation-and-governance.md.
    """
    if relative_path.endswith(".csv"):
        return _validate_csv_artifact(relative_path, artifact_path)
    if relative_path.endswith(".md"):
        return _validate_markdown_artifact(relative_path, artifact_path)
    return []


def _validate_optional_drl_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate an optional deep-RL artifact, adding DQN/PPO-specific content checks.

    Beyond the generic CSV/markdown checks, the optional group asserts that the comparison
    actually contrasts the three families that motivate the deep-RL bridge: the family
    comparison CSV must contain ``q_learning``, ``dqn``, and ``ppo`` rows; the bridge report
    must name both DQN and PPO; and the policy-gradient notes must mention the defining terms
    ("policy-gradient", "actor-critic", "ppo", "dqn").

    Args:
        relative_path: The artifact's contract-relative path (also selects its rules).
        artifact_path: Absolute/resolved path to the file on disk.

    Returns:
        Validation error messages for this artifact (empty if it passes).

    RL concept:
        Deep RL bridge -- value-based DQN vs actor-critic PPO vs tabular Q-learning.
        See docs/deep-rl.md and docs/policy-gradient-and-actor-critic.md.
    """
    if relative_path.endswith(".csv"):
        errors = _validate_csv_artifact(relative_path, artifact_path)
        if relative_path == "artifacts/drl_optional/rl_family_comparison.csv":
            rows = list(csv.DictReader(artifact_path.open(encoding="utf-8")))
            policies = {row["policy"] for row in rows}
            # Contract: the comparison must contrast all three families head-to-head.
            if {"q_learning", "dqn", "ppo"} - policies:
                errors.append(
                    "Optional DRL comparison must include q_learning, dqn, and ppo rows."
                )
        return errors
    if relative_path.endswith(".md"):
        errors = _validate_markdown_artifact(relative_path, artifact_path)
        content = artifact_path.read_text(encoding="utf-8")
        if relative_path == "artifacts/drl_optional/bridge_report.md":
            if "DQN" not in content or "PPO" not in content:
                errors.append("Optional DRL bridge report must mention both DQN and PPO.")
        if relative_path == "artifacts/drl_optional/policy_gradient_notes.md":
            # Required terms ensure the notes actually cover the policy-gradient/actor-critic arc.
            required_terms = ("policy-gradient", "actor-critic", "ppo", "dqn")
            lowered_content = content.lower()
            missing_terms = [term for term in required_terms if term not in lowered_content]
            if missing_terms:
                errors.append(
                    "Policy-gradient notes are missing required terms: "
                    + ", ".join(missing_terms)
                    + "."
                )
        return errors
    return []


def _validate_csv_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate a CSV artifact: required header columns present and at least one data row.

    The per-path ``required_columns`` table *is* the columnar contract for every tabular
    artifact (bandit traces, training curves, Q-tables, DP gap, policy comparisons, the
    deep-RL rollups). A file passes only if its header is non-empty, contains every required
    column for its path, and carries at least one data row. ``scripts/verify_artifacts.py``
    and ``tests/test_artifact_contract.py`` rely on exactly these rules.

    Args:
        relative_path: The artifact's contract-relative path; selects which columns are
            required.
        artifact_path: Path to the CSV file on disk.

    Returns:
        Validation error messages (empty if the CSV satisfies its contract).

    RL concept:
        Evaluation and governance -- machine-checkable columnar evidence.
        See docs/evaluation-and-governance.md.
    """
    required_columns = {
        "artifacts/concepts/concept_map.csv": (
            "concept",
            "showcase_component",
            "artifact",
        ),
        "artifacts/bandit/reward_trace.csv": (
            "step",
            "scenario_name",
            "context_signature",
            "action_label",
            "expected_reward",
            "optimal_action_label",
        ),
        "artifacts/bandit/regret_trace.csv": (
            "step",
            "scenario_name",
            "action_label",
            "optimal_action_label",
            "cumulative_regret",
        ),
        "artifacts/mdp/sample_episodes.csv": (
            "scenario_name",
            "week",
            "action",
            "reward",
            "next_risk",
        ),
        "artifacts/q_learning/training_curve.csv": (
            "episode",
            "scenario_id",
            "total_reward",
            "epsilon",
            "steps",
        ),
        "artifacts/q_learning/q_table.csv": (
            "week",
            "engagement",
            "completion",
            "pressure",
            "risk",
            "prior_interventions",
            "action",
            "q_value",
        ),
        "artifacts/dp/optimal_action_values.csv": (
            "week",
            "action",
            "optimal_q_value",
        ),
        "artifacts/dp/q_learning_gap.csv": (
            "week",
            "action",
            "learned_q_value",
            "optimal_q_value",
            "abs_gap",
        ),
        "artifacts/sarsa/training_curve.csv": (
            "episode",
            "scenario_id",
            "total_reward",
            "epsilon",
            "steps",
        ),
        "artifacts/sarsa/q_table.csv": (
            "week",
            "engagement",
            "completion",
            "pressure",
            "risk",
            "prior_interventions",
            "action",
            "q_value",
        ),
        "artifacts/policy_gradient/training_curve.csv": (
            "episode",
            "scenario_id",
            "total_reward",
            "baseline",
            "steps",
        ),
        "artifacts/eval/policy_comparison.csv": (
            "policy",
            "avg_reward",
            "avg_final_risk",
            "avg_intervention_cost",
            "solved_rate",
        ),
        "artifacts/eval/scenario_results.csv": (
            "policy",
            "scenario_id",
            "scenario_name",
            "total_reward",
            "final_risk",
            "actions",
        ),
        "artifacts/drl_optional/rl_family_comparison.csv": (
            "policy",
            "family",
            "avg_reward",
            "avg_final_risk",
            "solved_rate",
        ),
        "artifacts/drl_optional/scenario_rollups.csv": (
            "policy",
            "scenario_id",
            "scenario_name",
            "total_reward",
        ),
        "artifacts/drl_optional/training_summary.csv": (
            "policy",
            "step",
            "mean_reward",
            "mean_final_risk",
        ),
    }
    errors: list[str] = []
    with artifact_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [f"{relative_path} is empty or missing a header row."]
        missing_columns = [
            column
            for column in required_columns.get(relative_path, ())
            if column not in reader.fieldnames
        ]
        if missing_columns:
            errors.append(
                f"{relative_path} is missing required columns: {', '.join(missing_columns)}."
            )
        rows = list(reader)
        if not rows:
            errors.append(f"{relative_path} must contain at least one data row.")
    return errors


def _validate_markdown_artifact(relative_path: str, artifact_path: Path) -> list[str]:
    """Validate a markdown artifact: non-empty, heading-led, and (for the ladder) rung-complete.

    Generic rule: the file must be non-empty and start with a ``#`` heading. Path-specific
    rule: ``artifacts/concepts/algorithm_progression.md`` must mention each ladder rung --
    "q-learning", "dqn", "policy gradients", "actor-critic", and "ppo" -- so the progression
    narrative cannot silently skip a method.

    Args:
        relative_path: The artifact's contract-relative path; selects path-specific rules.
        artifact_path: Path to the markdown file on disk.

    Returns:
        Validation error messages (empty if the markdown satisfies its contract).

    RL concept:
        Evaluation and governance -- narrative evidence that covers the full ladder.
        See docs/evaluation-and-governance.md.
    """
    content = artifact_path.read_text(encoding="utf-8").strip()
    if not content:
        return [f"{relative_path} is empty."]
    if relative_path == "artifacts/concepts/algorithm_progression.md":
        # Contract: the ladder narrative must name every rung (no silently-skipped method).
        required_terms = ("q-learning", "dqn", "policy gradients", "actor-critic", "ppo")
        lowered_content = content.lower()
        missing_terms = [term for term in required_terms if term not in lowered_content]
        if missing_terms:
            return [
                "artifacts/concepts/algorithm_progression.md is missing: "
                + ", ".join(missing_terms)
                + "."
            ]
    if not content.startswith("#"):
        return [f"{relative_path} should start with a Markdown heading."]
    return []
