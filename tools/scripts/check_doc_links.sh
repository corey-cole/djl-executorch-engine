#!/usr/bin/env bash
# Verifies that every relative markdown link resolves to something that exists.
#
# With no arguments it checks every TRACKED .md file (git ls-files), so generated trees such as
# native/build-clangd/_deps and build/ are excluded for free. Pass explicit paths to check a
# subset -- that is how tools/tests/check_doc_links_test.sh drives it against fixtures.
#
# Only file existence is checked, not anchors: verifying #headings would mean parsing markdown
# heading-slug rules, and a wrong slug is a much cheaper mistake than a missing file.
#
# Code is stripped before extraction, and that is not optional -- both false positives it prevents
# occur in this repository today. A fenced block in host-buffer-contract-wip.md holds an ASan
# frame reading "operator new[](unsigned long)", and the design spec for this checker contains
# the literal `](...)` inside backticks while describing itself. Both parse as links otherwise.
set -euo pipefail

if [ "$#" -gt 0 ]; then
  files=("$@")
else
  cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  mapfile -t files < <(git ls-files '*.md')
fi

fail=0
for md in "${files[@]}"; do
  dir="$(dirname "$md")"
  while IFS= read -r target; do
    [ -n "$target" ] || continue
    case "$target" in
      http://* | https://* | mailto:* | \#*) continue ;;
    esac
    path="${target%%#*}"        # drop any #anchor; we check the file, not the heading
    [ -n "$path" ] || continue  # a bare "#anchor" left nothing behind
    if [ ! -e "${dir}/${path}" ]; then
      echo "broken link: ${md} -> ${target}"
      fail=1
    fi
  done < <(awk '/^[[:space:]]*```/ { inblock = !inblock; next } !inblock' "$md" \
             | sed -E 's/`[^`]*`//g' \
             | grep -oE '\]\([^)]+\)' | sed -E 's/^\]\(//; s/\)$//' | awk '{print $1}')
done

if [ "$fail" -ne 0 ]; then
  echo "check_doc_links: FAILED"
  exit 1
fi
echo "check_doc_links: all relative links resolve (${#files[@]} files)"
