#!/usr/bin/env python3
"""Export the assistant-side bridge contract, not learned model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.agent_bridge import action_mapping_markdown, policy_router_payload
from adaptive_course_assistant_rl.reporting import write_json_artifact, write_text_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    write_json_artifact(
        args.output_dir / "bridge" / "policy_router.json",
        policy_router_payload(),
    )
    write_text_artifact(
        args.output_dir / "bridge" / "action_mapping.md",
        action_mapping_markdown(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
