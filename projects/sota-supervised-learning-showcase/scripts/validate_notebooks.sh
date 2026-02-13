#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Validating notebooks (*.ipynb)..."

notebooks=()
while IFS= read -r notebook; do
  notebooks+=("${notebook}")
done < <(find "${ROOT_DIR}/notebooks" -maxdepth 1 -type f -name '*.ipynb' | sort)

if [ "${#notebooks[@]}" -eq 0 ]; then
  echo "No notebooks found in ${ROOT_DIR}/notebooks"
  exit 1
fi

for notebook in "${notebooks[@]}"; do
  uv run python - <<'PY' "${notebook}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

required_top_keys = {"cells", "metadata", "nbformat", "nbformat_minor"}
missing = required_top_keys - set(payload.keys())
if missing:
    raise SystemExit(f"{path}: missing keys {sorted(missing)}")

if not isinstance(payload["cells"], list):
    raise SystemExit(f"{path}: `cells` must be a list")

if payload["nbformat"] != 4:
    raise SystemExit(f"{path}: expected nbformat=4, found {payload['nbformat']}")

print(f"Notebook OK: {path.name} ({len(payload['cells'])} cells)")
PY
done

echo "Notebook validation complete."
