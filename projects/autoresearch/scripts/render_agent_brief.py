#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from autoresearch_showcase.agent_brief import render_agent_brief
from autoresearch_showcase.platforms import get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a Codex or Claude launch brief.")
    parser.add_argument("--platform", choices=["macos", "unix"], required=True)
    parser.add_argument("--agent", choices=["codex", "claude"], required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_profile(args.platform)
    content = render_agent_brief(profile, args.agent)
    if args.output is None:
        print(content)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
