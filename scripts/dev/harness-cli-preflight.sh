#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN_PATH="${HARNESS_PLAN_PATH:-$ROOT_DIR/plans/deep-learning-math-foundations-showcase.md}"
DESIGN_SPEC_PATH="${HARNESS_DESIGN_SPEC_PATH:-$ROOT_DIR/docs/superpowers/specs/2026-03-13-deep-learning-showcase-series-design.md}"

FAILURES=0

pass() {
  printf 'PASS: %s\n' "$1"
}

fail() {
  printf 'FAIL: %s\n' "$1"
  FAILURES=$((FAILURES + 1))
}

require_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    pass "file present: ${path#$ROOT_DIR/}"
  else
    fail "missing file: ${path#$ROOT_DIR/}"
  fi
}

require_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    pass "command available: $cmd"
  else
    fail "missing command: $cmd"
  fi
}

main() {
  require_cmd python3
  require_cmd uv
  require_cmd git
  require_cmd rg

  require_file "$ROOT_DIR/AGENTS.md"
  require_file "$ROOT_DIR/.codex/config.toml"
  require_file "$ROOT_DIR/.codex/agents/workflow_orchestrator.toml"
  require_file "$ROOT_DIR/.codex/harness/role-skill-matrix.toml"
  require_file "$ROOT_DIR/docs/agents/oodaris-harness-v2-operating-pack.md"
  require_file "$ROOT_DIR/docs/agents/harness-evals/README.md"
  require_file "$ROOT_DIR/scripts/harness_config_lint.py"
  require_file "$ROOT_DIR/scripts/dev/harness-cli-preflight.sh"
  require_file "$PLAN_PATH"
  require_file "$DESIGN_SPEC_PATH"

  if [[ "$FAILURES" -gt 0 ]]; then
    printf '\nHarness-lite preflight failed with %s issue(s).\n' "$FAILURES"
    exit 1
  fi

  printf '\nHarness-lite preflight passed.\n'
}

main
