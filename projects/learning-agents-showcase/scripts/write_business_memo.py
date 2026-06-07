#!/usr/bin/env python3
"""Write the deploy-shadow-reject memo from the current evaluation summary."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from learning_agents.reporting import recommendation_from_summary, write_text_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the business-memo runner's command-line flags.

    What + why: this is the governance hand-off, so its only job is to locate the artifacts tree
    that already holds the offline policy-comparison numbers. It exposes ``--output-dir`` and
    ``--quick``. There is no training loop here (the memo is derived from an existing CSV), so
    ``--quick`` is accepted only for a uniform CLI across the showcase runners and has no effect
    (it is discarded in ``main``).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        The deployment gate reads from offline evaluation rather than learning anything, so this
        runner takes no episode budget -- only the location of the prior results.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def read_policy_summary(path: Path) -> list[dict[str, str]]:
    """Load the policy-comparison CSV into a list of per-policy row dicts.

    What + why: reads the ``artifacts/eval/policy_comparison.csv`` summary written by the
    policy-evaluation runner; the returned rows are the offline evidence the deploy/shadow/reject
    recommendation is built from, so this runner must execute after that evaluation step.

    Args:
        path: Path to the ``policy_comparison.csv`` summary file.

    Returns:
        One dict per CSV row, mapping column name to string value (via ``csv.DictReader``).

    RL concept:
        These rows are the agent's report card -- per-policy reward and safety metrics gathered
        during offline evaluation, the inputs to the rollout decision.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """Derive the deploy/shadow/reject recommendation and write the business memo.

    What + why: this is where RL results meet human oversight. It reads the policy-comparison
    summary, asks ``recommendation_from_summary`` for a verdict and written rationale, and renders
    them into ``artifacts/business/deploy_shadow_reject_memo.md`` -- turning offline numbers into a
    single, reviewable business recommendation. It consumes ``eval/policy_comparison.csv``, so the
    policy-evaluation runner must run first (the Makefile orders them).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Governance hand-off from offline evaluation to a deployment decision: a learned policy
        earns a guarded rollout only when it is both safe enough and better than the incumbent
        heuristic router; otherwise it is shadowed or rejected.
    """
    args = parse_args(argv)
    del args.quick  # memo is derived from an existing CSV; the --quick flag does not apply here
    summary_path = args.output_dir / "eval" / "policy_comparison.csv"
    rows = read_policy_summary(summary_path)
    recommendation, rationale = recommendation_from_summary(rows)
    memo = (
        "# Deploy, Shadow, or Reject Memo\n\n"
        f"Recommendation: {recommendation}.\n\n"
        "## Why\n\n"
        f"{rationale}\n\n"
        "## What This Means\n\n"
        "- `deploy`: rare in this teaching repo and only appropriate when offline risk is low.\n"
        "- `shadow`: collect more evidence with human review and no automated actioning.\n"
        "- `reject`: redesign the reward, policy, or safety controls before moving further.\n"
    )
    write_text_artifact(args.output_dir / "business" / "deploy_shadow_reject_memo.md", memo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
