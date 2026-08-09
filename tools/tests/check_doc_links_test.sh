#!/usr/bin/env bash
# Exercises check_doc_links.sh against fixtures: a broken link must fail and be named,
# a resolvable one must pass, and non-file targets must be ignored.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKER="${REPO_ROOT}/tools/scripts/check_doc_links.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail() { echo "FAIL: $1"; exit 1; }

# Case 1: a broken relative link must fail, and the output must name the target.
touch "$tmp/real.md"
printf '[ok](real.md) and [bad](missing.md)\n' > "$tmp/doc.md"
if out="$("$CHECKER" "$tmp/doc.md" 2>&1)"; then
  fail "expected non-zero exit for a broken link, got success: $out"
fi
grep -q "missing.md" <<<"$out" || fail "output did not name the broken target: $out"

# Case 2: a file whose links all resolve must pass.
printf '[ok](real.md)\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "a resolvable link must pass"

# Case 3: URLs, mailto and bare anchors are not files and must be ignored.
printf '[a](https://example.com) [b](http://example.com) [c](mailto:x@y.z) [d](#section)\n' \
  > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "non-file targets must be ignored"

# Case 4: an anchor suffix on a real file is fine; only file existence is checked.
printf '[e](real.md#some-heading)\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "an anchor on an existing file must pass"

# Case 5: a link to a directory resolves (docs/research/ style links).
mkdir -p "$tmp/subdir"
printf '[f](subdir)\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "a directory target must resolve"

# Case 6: a markdown link title must not be mistaken for part of the path.
printf '[g](real.md "Some Title")\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "a link with a title must resolve"

# Case 7: an inline code span is not a link. Regression fixture: the design spec for this very
# work contains the literal `](...)` inside backticks while describing this checker.
printf 'the checker extracts `](target)` pairs\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "inline code spans must be ignored"

# Case 8: a fenced code block is not a link. Regression fixture: host-buffer-contract-wip.md
# contains an ASan stack frame reading "operator new[](unsigned long)".
printf 'before\n```\n#0 operator new[](unsigned long)\n```\nafter\n' > "$tmp/doc.md"
"$CHECKER" "$tmp/doc.md" >/dev/null 2>&1 || fail "fenced code blocks must be ignored"

echo "PASS"
