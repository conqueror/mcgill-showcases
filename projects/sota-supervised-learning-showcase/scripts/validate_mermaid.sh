#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
OUT_DIR="${ROOT_DIR}/artifacts/diagrams"

mkdir -p "${OUT_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

FILES=("${ROOT_DIR}/README.md")
while IFS= read -r file; do
  FILES+=("${file}")
done < <(find "${ROOT_DIR}/docs" -maxdepth 1 -name '*.md' -type f | sort)

index=0
for markdown_file in "${FILES[@]}"; do
  awk -v out="${TMP_DIR}" -v prefix="$(basename "${markdown_file}" .md)" '
    BEGIN { in_block = 0; idx = 0 }
    /^```mermaid[[:space:]]*$/ {
      in_block = 1
      idx += 1
      out_file = sprintf("%s/%s_%03d.mmd", out, prefix, idx)
      next
    }
    /^```[[:space:]]*$/ && in_block == 1 {
      in_block = 0
      next
    }
    in_block == 1 {
      print $0 >> out_file
    }
  ' "${markdown_file}"
done

diagrams=()
while IFS= read -r diagram_file; do
  diagrams+=("${diagram_file}")
done < <(find "${TMP_DIR}" -name '*.mmd' -type f | sort)

if [ "${#diagrams[@]}" -eq 0 ]; then
  echo "No Mermaid blocks found."
  exit 1
fi

echo "Found ${#diagrams[@]} Mermaid diagrams. Rendering..."
for diagram in "${diagrams[@]}"; do
  index=$((index + 1))
  base_name="$(basename "${diagram}" .mmd)"
  output_svg="${OUT_DIR}/${base_name}.svg"
  npx -y @mermaid-js/mermaid-cli -i "${diagram}" -o "${output_svg}" >/dev/null
  echo "[${index}/${#diagrams[@]}] ${base_name}.svg"
done

echo "Mermaid validation complete. Output SVG files in ${OUT_DIR}."
