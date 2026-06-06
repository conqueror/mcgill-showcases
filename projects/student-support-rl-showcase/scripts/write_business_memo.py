#!/usr/bin/env python3
"""Write the deploy-shadow-reject memo from the current evaluation summary.

This is the governance hand-off: it turns the offline policy-comparison numbers into a single
business recommendation (deploy / shadow / reject) with a written rationale, the step where RL
results meet human oversight. It consumes the evaluation summary produced by the policy-evaluation
runner, so that script must run first. See docs/evaluation-and-governance.md.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from student_support_rl.reporting import recommendation_from_summary, write_text_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the business-memo runner's command-line flags.

    Exposes ``--output-dir`` and ``--quick``. There is no training loop here (the memo is derived
    from an existing CSV), so ``--quick`` is accepted for a uniform CLI but has no effect (it is
    discarded in ``main``).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def read_policy_summary(path: Path) -> list[dict[str, str]]:
    """Load the policy-comparison CSV into a list of per-policy row dicts.

    Reads the ``artifacts/eval/policy_comparison.csv`` summary written by the policy-evaluation
    runner; the returned rows are the offline evidence the deploy/shadow/reject recommendation is
    built from.

    Args:
        path: Path to the ``policy_comparison.csv`` summary file.

    Returns:
        One dict per CSV row, mapping column name to string value (via ``csv.DictReader``).
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """Derive the deploy/shadow/reject recommendation and write the business memo.

    Reads the policy-comparison summary, asks ``recommendation_from_summary`` for a verdict and
    rationale, and writes ``artifacts/business/deploy_shadow_reject_memo.md``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Governance hand-off from offline evaluation to a deployment decision. See
        docs/evaluation-and-governance.md.
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
