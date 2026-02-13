#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT_DIR/artifacts/figures/mermaid"
README_OUT_DIR="$OUT_DIR/readme"

mkdir -p "$OUT_DIR"
mkdir -p "$README_OUT_DIR"

for diagram in "$ROOT_DIR"/docs/diagrams/*.mmd; do
  name="$(basename "$diagram" .mmd)"
  out_file="$OUT_DIR/${name}.svg"
  echo "Rendering ${name}.mmd -> ${out_file}"
  npx -y @mermaid-js/mermaid-cli -i "$diagram" -o "$out_file" >/dev/null
done

echo "Rendering Mermaid blocks from README.md"
ROOT_DIR="$ROOT_DIR" python - <<'PY'
from pathlib import Path
import subprocess
import tempfile
import os

root = Path(os.environ["ROOT_DIR"])
readme = (root / "README.md").read_text(encoding="utf-8")
out_dir = root / "artifacts/figures/mermaid/readme"

blocks: list[str] = []
inside = False
buffer: list[str] = []
for line in readme.splitlines():
    if line.strip() == "```mermaid":
        inside = True
        buffer = []
        continue
    if inside and line.strip() == "```":
        inside = False
        if buffer:
            blocks.append("\n".join(buffer) + "\n")
        continue
    if inside:
        buffer.append(line)

for stale in out_dir.glob("readme_*.svg"):
    stale.unlink()

with tempfile.TemporaryDirectory(prefix="mermaid_readme_") as tmp:
    tmp_dir = Path(tmp)
    for idx, block in enumerate(blocks, start=1):
        src = tmp_dir / f"readme_{idx}.mmd"
        dst = out_dir / f"readme_{idx}.svg"
        src.write_text(block, encoding="utf-8")
        subprocess.run(
            [
                "npx",
                "-y",
                "@mermaid-js/mermaid-cli",
                "-i",
                str(src),
                "-o",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

print(f"Rendered {len(blocks)} Mermaid blocks from README.md")
PY

echo "Mermaid validation complete. Rendered diagrams are in $OUT_DIR"
