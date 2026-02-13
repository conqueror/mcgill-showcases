#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

check_links_in_file() {
  local markdown_file="$1"
  local file_dir
  file_dir="$(cd "$(dirname "${markdown_file}")" && pwd)"

  local links
  links="$(grep -oE '\[[^]]+\]\([^)]+\)' "${markdown_file}" || true)"
  if [ -z "${links}" ]; then
    return 0
  fi

  while IFS= read -r match; do
    [ -z "${match}" ] && continue
    local target
    target="$(printf '%s' "${match}" | sed -E 's/^[^[]*\[[^]]+\]\(([^)]+)\)$/\1/')"

    if [[ "${target}" == \#* ]]; then
      continue
    fi

    case "${target}" in
      http://*|https://*|mailto:*|javascript:*)
        continue
        ;;
    esac

    local path_only="${target%%#*}"
    path_only="${path_only%%\?*}"
    [ -z "${path_only}" ] && continue

    local candidate
    if [[ "${path_only}" = /* ]]; then
      candidate="${path_only}"
    else
      candidate="${file_dir}/${path_only}"
    fi

    if [ ! -e "${candidate}" ]; then
      echo "Broken link in ${markdown_file}: ${target}"
      return 1
    fi
  done <<< "${links}"
}

echo "Validating local Markdown links..."

docs_and_readme=("${ROOT_DIR}/README.md")
while IFS= read -r file; do
  docs_and_readme+=("${file}")
done < <(find "${ROOT_DIR}/docs" -maxdepth 1 -type f -name '*.md' | sort)

for markdown_file in "${docs_and_readme[@]}"; do
  check_links_in_file "${markdown_file}"
done

echo "Markdown link validation complete."
