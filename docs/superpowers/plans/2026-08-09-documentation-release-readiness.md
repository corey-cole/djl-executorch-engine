# Documentation Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project's documentation fit to publish — a README that explains what the engine offers and how to use it, zero javadoc warnings, a documented native layer, and a `docs/` directory where current guidance is distinguishable from historical research.

**Architecture:** Content work, not code, with three mechanical gates so it cannot rot: a link checker wired into `check`, a quickstart that is a compiled class rather than a snippet, and a zero-warning javadoc baseline. Files move but nothing is deleted; every moved document gains a status header and its inbound references are updated in the same commit.

**Tech Stack:** Markdown, Javadoc (JDK 17), Gradle Kotlin DSL, Bash, C++ comments.

**Spec:** `docs/superpowers/specs/2026-08-09-documentation-release-readiness-design.md`

## Global Constraints

- **No emoji anywhere in `README.md`.** Not in headings, status markers, table cells, or callouts. Emphasis uses bold; admonitions use a `> **Note:**` blockquote. This is a constraint to hold, not a cleanup — `README.md` and `CLAUDE.md` are emoji-free today.
- **The no-emoji rule applies to `README.md` only.** Research documents use `⚠` as a load-bearing marker in their own text (`handover-to-engine-2.md` says "The only ⚠ is C8" and keys a table column off it). They move **unedited apart from the status header**.
- **Nothing is deleted.** Every moved document keeps its content.
- **Triage rule:** a document stays in `docs/` if something *outside itself* cites it as current — a code comment, CLAUDE.md, or the README. Otherwise it moves to `docs/research/`. Size is not the signal.
- **Inbound references are updated in the same commit as the move.** A move that leaves a dangling link is an incomplete task.
- **No Doxygen**, no generated C++ documentation site, no comment-syntax conversion. Native comments stay in the prose-explaining-*why* style that `native/core/et_runtime.h` already uses.
- **No javadoc failure gate.** Doclint warnings stay non-fatal; this is content work only. Target is zero warnings, not a new build rule.
- **No rewriting of `benchmarking.md`, `ci-native-build.md`, or `executorch-build-notes.md`.** They are current and cited; they stay as they are.
- Markdown wraps at 100 columns to match the existing docs.

---

### Task 1: Doc link checker

Built first because Tasks 2, 4, 5 and 8 all rely on it. Eight files move and the README's internal links are rewritten, so a dangling link is the most likely defect in this whole cycle.

**Files:**
- Create: `tools/scripts/check_doc_links.sh`
- Create: `tools/tests/check_doc_links_test.sh`
- Modify: `build.gradle.kts` (new task, wired into `check`)

**Interfaces:**
- Consumes: nothing.
- Produces: `tools/scripts/check_doc_links.sh [file...]` — exits non-zero and prints `broken link: <file> -> <target>` for each unresolvable relative link. With no arguments it checks every tracked `.md` file. Gradle task `checkDocLinks`.

- [ ] **Step 1: Write the failing test**

Create `tools/tests/check_doc_links_test.sh`:

```bash
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
```

Make it executable: `chmod +x tools/tests/check_doc_links_test.sh`

- [ ] **Step 2: Run the test to verify it fails**

Run: `./tools/tests/check_doc_links_test.sh`

Expected: FAIL — `check_doc_links.sh: No such file or directory`.

- [ ] **Step 3: Write the checker**

Create `tools/scripts/check_doc_links.sh`:

```bash
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
```

The pipeline reads right to left: the first `awk` drops fenced code blocks, `sed` drops inline
code spans, `grep` extracts `](target)` pairs, and the trailing `awk '{print $1}'` drops markdown
link titles so `[x](file.md "Title")` yields `file.md`. Only ``` fences are handled, not `~~~`;
no document here uses the latter.

Make it executable: `chmod +x tools/scripts/check_doc_links.sh`

- [ ] **Step 4: Run the test to verify it passes**

Run: `./tools/tests/check_doc_links_test.sh`

Expected: `PASS`.

- [ ] **Step 5: Run it over the real tree**

Run: `./tools/scripts/check_doc_links.sh`

Expected: `check_doc_links: all relative links resolve (41 files)`. This was verified against the tree at the time of writing — the baseline is clean, so **any** hit here means either the script was mistyped or a link broke since. Do not "fix" a reported link without first confirming it is a real link rather than a code fragment.

- [ ] **Step 6: Wire it into Gradle**

In `build.gradle.kts`, after the `statsDegradedTest` registration:

```kotlin
val checkDocLinks = tasks.register<Exec>("checkDocLinks") {
    group = "verification"
    description = "Verifies every relative markdown link in tracked .md files resolves."
    commandLine("tools/scripts/check_doc_links.sh")
    // Skipped on Windows: Exec cannot run a .sh directly there, and the check is
    // platform-independent, so the Linux legs of CI already cover it for everyone.
    onlyIf { !System.getProperty("os.name").lowercase().contains("windows") }
}
```

Then add it to the existing `tasks.check` block alongside the other entries:

```kotlin
tasks.check {
    dependsOn(
        tasks.named("intraOpTest"),
        jmxDisabledTest,
        statsDegradedTest,
        checkDocLinks,
    )
}
```

- [ ] **Step 7: Verify the Gradle task runs**

Run: `./gradlew checkDocLinks`

Expected: `BUILD SUCCESSFUL` with `check_doc_links: all relative links resolve`.

- [ ] **Step 8: Commit**

```bash
git add tools/scripts/check_doc_links.sh tools/tests/check_doc_links_test.sh build.gradle.kts
git commit -m "test: add a doc link checker wired into check"
```

---

### Task 2: Triage docs/ and commit the untracked design records

**Files:**
- Create: `docs/research/` (directory)
- Create: `docs/README.md`
- Move: seven documents plus one directory into `docs/research/`
- Modify: `docs/benchmarking.md`, `native/harness/et_overread_harness.cpp`, `CLAUDE.md` (inbound references)
- Add: eight untracked `docs/superpowers/` files

**Interfaces:**
- Consumes: `tools/scripts/check_doc_links.sh` (Task 1).
- Produces: the final `docs/` layout that Tasks 4, 5, 8 and 9 link into.

- [ ] **Step 1: Record the pre-move link baseline**

```bash
./tools/scripts/check_doc_links.sh
```

Expected: passes. Anything broken now must be fixed before moving files, or you cannot tell new breakage from old.

- [ ] **Step 2: Move the research documents**

```bash
mkdir -p docs/research
git mv docs/handover-to-engine-2.md docs/research/
git mv docs/handover-windows-static-cxx17-findings.md docs/research/
git mv docs/host-buffer-contract-wip.md docs/research/
mv docs/handover-to-engine.md docs/research/
mv docs/handover-windows-static-cxx17.md docs/research/
mv docs/panama-research-sketch.md docs/research/
mv docs/iree-lessons-learned docs/research/
```

Two of these are tracked and use `git mv`; four are untracked working files and use plain `mv`. They are added in Step 5.

`executorch-host-buffer-contract-brief.md` **stays in `docs/`** — CLAUDE.md cites it for the input-copy contract and `native/harness/et_overread_harness.cpp:3` cites it by section (`§3/W4`). It is current reference despite its size.

- [ ] **Step 3: Add a status header to each moved document**

Insert at the very top of each moved file, above its existing title, filling in the two bracketed fields from the file's own content and `git log`:

```markdown
> **Historical record — not current guidance.**
> Written [date]. Kept for the reasoning it captures; details may have been superseded by later
> work. For current guidance see [docs/README.md](../README.md).
```

For `docs/research/panama-research-sketch.md` use this instead, because it is open research rather than completed work:

```markdown
> **Open research — no decision made, nothing implemented.**
> Written 2026-08. Explores a possible direction; it is not a plan and nothing here is committed to.
> For current guidance see [docs/README.md](../README.md).
```

For `docs/research/iree-lessons-learned/`, add the historical header to each `.md` file inside, with `../../README.md` as the link target since they sit one level deeper.

- [ ] **Step 4: Fix the inbound references the move broke**

Three documents are cited from outside themselves, and `host-buffer-contract-wip.md` has links of its own that break by descending a level.

In `docs/benchmarking.md`, the reference to `executorch-host-buffer-contract-brief.md` is unchanged — that file did not move. Verify no edit is needed there.

In `docs/research/host-buffer-contract-wip.md`, its own links to the contract brief now need to climb one level. Replace every occurrence of `docs/executorch-host-buffer-contract-brief.md` with `../executorch-host-buffer-contract-brief.md`:

```bash
sed -i 's|docs/executorch-host-buffer-contract-brief\.md|../executorch-host-buffer-contract-brief.md|g' \
  docs/research/host-buffer-contract-wip.md
```

Then find and update every remaining reference to a moved file:

```bash
grep -rIn "docs/handover-to-engine\|docs/handover-windows-static\|docs/host-buffer-contract-wip\|docs/panama-research-sketch\|docs/iree-lessons-learned" \
  --exclude-dir=.git --exclude-dir=build --exclude-dir=build-clangd --exclude-dir=_deps .
```

Rewrite each hit to the `docs/research/...` path. Expect one hit each for `handover-to-engine-2`, `handover-windows-static-cxx17-findings`, and `iree-lessons-learned`.

- [ ] **Step 5: Write the docs index**

Create `docs/README.md`:

```markdown
# Documentation

Current reference material lives here. Point-in-time records live in
[research/](research), which keeps the reasoning behind past decisions without presenting it as
current guidance.

## Current

| Document | What it covers |
|---|---|
| [building.md](building.md) | Building the native shim and running the test suites from source |
| [native-architecture.md](native-architecture.md) | The C++ layer: core/JNI split, ownership, staging, process-global state |
| [benchmarking.md](benchmarking.md) | The timing harness and how to read its output |
| [ci-native-build.md](ci-native-build.md) | How the native build matrix works in CI |
| [executorch-build-notes.md](executorch-build-notes.md) | Building an ExecuTorch runtime from source (escape hatch) |
| [executorch-host-buffer-contract-brief.md](executorch-host-buffer-contract-brief.md) | The host-buffer contract: when ExecuTorch copies an input and when it borrows |

Design specs and implementation plans are under [superpowers/](superpowers).

## Research and historical records

[research/](research) holds completed work orders, superseded working notes, and open research.
Each file carries a header saying what it is and when it was written. Nothing there is current
guidance; read it for reasoning, not instructions.
```

Note this index links `building.md` and `native-architecture.md`, which Tasks 5 and 8 create. Until then `checkDocLinks` will fail on those two rows — expected, and the reason this task's own link check (Step 7) passes explicit files rather than the whole tree.

- [ ] **Step 6: Add the untracked design records**

Eight `docs/superpowers/` specs and plans were never committed while 29 siblings were. They record decisions that shipped, so they are added as-is:

```bash
git add docs/superpowers/plans/2026-06-30-etruntime-core-extraction.md \
        docs/superpowers/plans/2026-06-30-executorch-logging-bridge.md \
        docs/superpowers/plans/2026-07-01-native-timing-harness.md \
        docs/superpowers/plans/2026-07-03-engine-runtime-consumption.md \
        docs/superpowers/specs/2026-06-30-etruntime-core-extraction-design.md \
        docs/superpowers/specs/2026-06-30-executorch-logging-bridge-design.md \
        docs/superpowers/specs/2026-07-01-native-timing-harness-design.md \
        docs/superpowers/specs/2026-07-02-executorch-runtime-dist-design.md
```

- [ ] **Step 7: Verify the moved files' links resolve**

```bash
./tools/scripts/check_doc_links.sh $(git ls-files 'docs/research/*.md' 'docs/research/**/*.md')
```

Expected: passes. This checks the moved files specifically; the whole-tree check comes back green in Task 9 once `building.md` and `native-architecture.md` exist.

- [ ] **Step 8: Commit**

```bash
git add -A docs/ native/ CLAUDE.md
git commit -m "docs: separate current reference from research records"
```

---

### Task 3: A quickstart that cannot rot

The README's most-read code must be compiled, not transcribed. A snippet that silently stops matching the API is worse than no snippet.

**Files:**
- Create: `example/src/main/java/org/measly/example/QuickStart.java`
- Modify: `example/build.gradle.kts` (a run task)

**Interfaces:**
- Consumes: nothing.
- Produces: `org.measly.example.QuickStart` with `main(String[])`, quoted verbatim by the README in Task 4. Gradle task `:example:runQuickStart`.

- [ ] **Step 1: Write the quickstart**

Create `example/src/main/java/org/measly/example/QuickStart.java`:

```java
package org.measly.example;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * The smallest useful ExecuTorch-under-DJL program: load a {@code .pte} and run one prediction.
 *
 * <p>This class is the source of the README's quickstart. It is compiled by the build, so an API
 * change breaks CI rather than leaving the README quietly wrong. Keep the two in sync.
 *
 * <p>Defaults to the two-input float32 {@code add} model committed at {@code native/spike/add.pte},
 * so it runs from a fresh clone with no model export step. Pass a directory and model name to run
 * a different {@code .pte}.
 */
public final class QuickStart {

    private QuickStart() {}

    /** Turns a {@code float[]} into the model's input list and its output back into a float. */
    private static final class AddTranslator implements Translator<float[], Float> {
        @Override
        public NDList processInput(TranslatorContext ctx, float[] input) {
            // One NDArray per model input. The add model takes two 1-element float32 tensors.
            NDArray a = ctx.getNDManager().create(new float[] {input[0]});
            NDArray b = ctx.getNDManager().create(new float[] {input[1]});
            return new NDList(a, b);
        }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow().toFloatArray()[0];
        }
    }

    /**
     * Runs one prediction and prints the result.
     *
     * @param args optionally the model directory and model name; defaults to {@code
     *     native/spike} and {@code add}
     * @throws Exception if the model cannot be loaded or the prediction fails
     */
    public static void main(String[] args) throws Exception {
        Path modelDir = Paths.get(args.length > 0 ? args[0] : "native/spike");
        String modelName = args.length > 1 ? args[1] : "add";

        Criteria<float[], Float> criteria =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optEngine("ExecuTorch") // this engine, by name
                        .optModelPath(modelDir)
                        .optModelName(modelName)
                        .optTranslator(new AddTranslator())
                        .build();

        // One ZooModel and one Predictor per thread: forward() is not safe to share.
        try (ZooModel<float[], Float> model = criteria.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
            System.out.println("2 + 3 = " + predictor.predict(new float[] {2f, 3f}));
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `./gradlew :example:compileJava`

Expected: `BUILD SUCCESSFUL`. This is the gate that keeps the README honest.

- [ ] **Step 3: Add a run task**

In `example/build.gradle.kts`, after the existing `application { ... }` block:

```kotlin
// The README quickstart, runnable. workingDir is the repo root so the default model path
// (native/spike/add.pte) resolves from a fresh clone with no export step.
tasks.register<JavaExec>("runQuickStart") {
    group = "application"
    description = "Runs the README quickstart against native/spike/add.pte."
    mainClass = "org.measly.example.QuickStart"
    classpath = sourceSets["main"].runtimeClasspath
    workingDir = rootProject.projectDir
}
```

- [ ] **Step 4: Run it end to end**

Run: `./gradlew :example:runQuickStart`

Expected: prints `2 + 3 = 5.0`.

If it fails with a missing-library error, the native shim is not staged. Build it first with `./native/build.sh`, then re-run.

- [ ] **Step 5: Commit**

```bash
git add example/src/main/java/org/measly/example/QuickStart.java example/build.gradle.kts
git commit -m "docs: add a compiled quickstart for the README"
```

---

### Task 4: Rewrite the README as a landing page

**Files:**
- Modify: `README.md` (full rewrite)

**Interfaces:**
- Consumes: `QuickStart.java` (Task 3), `docs/building.md` (Task 5 — write the link now, the file lands in Task 5), the `docs/` layout (Task 2).
- Produces: the landing page. Nothing depends on it.

- [ ] **Step 1: Rewrite README.md**

Replace the whole file with these sections, in this order. Reuse existing prose verbatim where the current README already says it well — the platform table, the dependency snippets, and the third-party licence table are all kept as they are.

1. **Title and what it is.** Two short paragraphs: DJL 0.36.0 supports only the deprecated TorchScript export API; this engine adds ExecuTorch as a separate DJL engine so models exported with the newer backend run under DJL, and so a codebase can migrate off TorchScript gradually. State plainly that it is CPU-only with limited NDArray support.

2. **Supported platforms.** The existing table from the current README, unchanged.

3. **Add the dependency.** The existing Gradle capability block and Maven classifier block, unchanged, plus the existing sentence about swapping the platform.

4. **Quickstart.** Introduce it in one sentence, then quote `QuickStart.java` — the `AddTranslator` class and the body of `main`. Add, after the snippet:

   ```markdown
   The full file is [`example/src/main/java/org/measly/example/QuickStart.java`](example/src/main/java/org/measly/example/QuickStart.java);
   run it with `./gradlew :example:runQuickStart`.
   ```

5. **Configuration and tuning.** Three subsections:
   - `ai.djl.executorch.num_threads` — sizes the intra-op (XNNPACK) pool, process-global and write-once, sealed at the first model load; effective value from `EtEngine.getIntraOpThreads()`.
   - `workspaceSharingMode` — per model via `Criteria.optOption`, or `ai.djl.executorch.workspace_sharing_mode` as the JVM-wide default; values `disabled`, `per_model`, `global`.
   - **Threading**, as a `> **Note:**` blockquote: more caller threads is usually *slower* under the default `global` sharing mode, because XNNPACK already parallelises inside one `forward()` and concurrent delegate calls serialise on a process-global workspace mutex. Include the measured figures from CLAUDE.md: 4-core/8-thread host, MobileNetV2, 1 thread 462 forwards/s, 4 threads 305, 8 threads 147, peak RSS 33 MB to 224 MB. Then the counterpoint: under `disabled`, achieved parallelism at one intra-op thread was 1.12/2.23/4.35/7.13 at 1/2/4/8 caller threads, versus 1.12/1.12/1.12/1.17 under `global`.

6. **Monitoring.** `EtEngineStats.snapshot()` with a three-line usage example; the JMX object name `org.measly.executorch:type=EtEngineStats`, auto-registered at the first model load; `ai.djl.executorch.jmx_enabled=false` to opt out; and the convention that byte fields use `-1` for unavailable and `0` for genuinely zero — `stagingBytes` is legitimately `0` for memory-planned models, which is nearly all of them.

7. **Limitations.** One `Model`/`Predictor` per thread and never close a model with a forward in flight; XNNPACK weight cache deliberately not exposed; the XNNPACK delegate workspace is not included in the reported native footprint.

8. **Building from source.** Three sentences plus `See [docs/building.md](docs/building.md).`

9. **Third-party licenses.** The existing section verbatim — it is a legal notice tied to the runtime pin.

- [ ] **Step 2: Verify no emoji slipped in**

```bash
grep -nP "[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{2B00}-\x{2BFF}\x{FE0F}]" README.md && \
  echo "EMOJI FOUND - remove them" || echo "clean: no emoji"
```

Expected: `clean: no emoji`.

- [ ] **Step 3: Verify the quickstart snippet matches the compiled source**

Read `example/src/main/java/org/measly/example/QuickStart.java` and compare it line by line against the README snippet. They must agree exactly on class names, method signatures, and the `Criteria` builder chain. A snippet that has drifted from the file defeats the point of Task 3.

- [ ] **Step 4: Check links**

Run: `./tools/scripts/check_doc_links.sh README.md`

Expected: passes. The `docs/building.md` link resolves only after Task 5; if it fails on that one row alone, continue and confirm in Task 9.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as a consumer landing page"
```

---

### Task 5: docs/building.md and the .clangd repoint

**Files:**
- Create: `docs/building.md`
- Modify: `.clangd` (comment header)

**Interfaces:**
- Consumes: the `docs/` layout (Task 2).
- Produces: `docs/building.md`, linked from `README.md` and `docs/README.md`.

- [ ] **Step 1: Write docs/building.md**

Move the build content out of the old README (recover it with `git show HEAD~1:README.md` if Task 4 has already landed) and organise it as:

1. **Prerequisites** — Docker for the Linux build; JDK 17 for Gradle; no ExecuTorch checkout needed, but network access is required for the runtime tarball and Catch2.
2. **The glibc floor, and why the container is not optional** — ExecuTorch 1.3 pins `torch==2.12.0`, whose wheel needs glibc ≥ 2.28, so the shipped `.so` must be built in `manylinux_2_28`. A host build breaks the floor: fine for local `./gradlew test`, never for a release.
3. **Building the native shim** — `./native/local_build_wrapper.sh` as the blessed path, `./native/build.sh` as the local fast path with the do-not-ship warning, and `ET_INSTALL` as the escape hatch.
4. **Windows** — no container; MSVC 2022 or later with the C++ toolchain, discovered edition-agnostically via `vswhere`; Ninja and CMake on PATH; Git-Bash invoked by explicit path with a non-login shell; a JDK for headers only. State the `CMAKE_BUILD_TYPE=Release` CRT constraint and that `native/tests/check_windows_crt.sh` is the real gate because MSVC does not reliably diagnose a mismatch.
5. **Running the tests** — `./gradlew test`, `leakTest`, `build`; the note that JVM integration tests load the native library so the shim must be built and staged first.
6. **Native QA and benchmarking** — `build_qa.sh`, `bench.sh`, `build_variants.sh`, all through the container wrapper.
7. **Container file ownership** — which scripts leave root-owned directories and the `sudo chown` fix.
8. **Verifying runtime provenance** — the `gh attestation verify` command.
9. **Editor setup (clangd)** — this is the section `.clangd` has always pointed at and which never existed:

   ```markdown
   ## Editor setup (clangd)

   `.clangd` points at `native/build-clangd`, a compile database that no build script touches.
   Generate or refresh it with:

   ```bash
   ./native/gen_clangd_db.sh
   ```

   It runs two CMake configures and merges them, because `native/CMakeLists.txt` builds the JNI
   shim only when `ET_BUILD_QA` and `ET_BUILD_BENCH` are both off — so no single configure covers
   both `jni/` and `test/`. Configure only; nothing is compiled.

   Re-run it after bumping `native/cmake/EtRuntimePin.cmake` or changing compile flags. The
   database is refreshed only by that script, so a stale one keeps resolving against the previous
   runtime's headers, silently and with no warning.
   ```

- [ ] **Step 2: Repoint .clangd at the real section**

In `.clangd`, the comment header currently references a README section that does not exist. Replace that reference with:

```
#   ./native/gen_clangd_db.sh          (see "Editor setup (clangd)" in docs/building.md)
```

Verify no other reference to the phantom README section survives:

```bash
grep -rn "Editor setup" --include="*.clangd" --include="*.md" --include="*.sh" . | grep -v docs/building.md
```

Expected: no output.

- [ ] **Step 3: Check links**

Run: `./tools/scripts/check_doc_links.sh docs/building.md docs/README.md README.md`

Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add docs/building.md .clangd
git commit -m "docs: move build instructions to docs/building.md"
```

---

### Task 6: Javadoc to zero warnings

32 warnings: 25 in the published module across 8 files, 7 in `example/`. The published module's javadoc jar already ships to Maven Central, so these are user-facing today.

**Files:**
- Modify: `src/main/java/org/measly/executorch/jni/EtNative.java`, `EtTensor.java`, `EtMethodMeta.java`
- Modify: `src/main/java/org/measly/executorch/translate/DType.java`, `MapTranslator.java`
- Modify: `src/main/java/org/measly/executorch/engine/EtEngine.java`, `EtDataTypes.java`, `LibUtils.java`
- Modify: `example/src/main/java/org/measly/example/MobilenetExample.java`, `ModelArtifacts.java`, `Variant.java`

**Interfaces:**
- Consumes: nothing.
- Produces: a zero-warning `./gradlew javadoc`.

- [ ] **Step 1: Record the baseline**

```bash
./gradlew :javadoc :example:javadoc --rerun-tasks 2>&1 | grep -cE "^/home.*warning"
```

Expected: `32`.

- [ ] **Step 2: Fix `EtTensor.java` (4 warnings — lines 7, 8, 9, 11)**

Add javadoc to the three public fields and the constructor:

```java
public final class EtTensor {
    /** Tensor dimensions, outermost first. */
    public final long[] shape;

    /** ExecuTorch {@code ScalarType} integer code; see {@code EtDataTypes} for the mapping. */
    public final int scalarType;

    /**
     * The tensor's raw bytes in native order. Inputs are direct buffers, which the native side
     * reads without copying; outputs are heap buffers holding a single copy out of ExecuTorch's
     * arena, because the arena's contents are invalidated by the next forward.
     */
    public final ByteBuffer data;

    /**
     * @param shape tensor dimensions, outermost first
     * @param scalarType ExecuTorch {@code ScalarType} integer code
     * @param data raw bytes in native order; direct for an input, heap for an output
     */
    public EtTensor(long[] shape, int scalarType, ByteBuffer data) {
```

- [ ] **Step 3: Fix `DType.java` (5 warnings — lines 9, 15, 21, 31, 58)**

Add a javadoc line to each enum constant and to the public `from` method:

```java
public enum DType {
    /** IEEE-754 single precision. */
    FLOAT32 {
    ...
    /** IEEE-754 double precision. */
    FLOAT64 {
    ...
    /** 32-bit signed integer; a value outside its range is rejected rather than truncated. */
    INT32 {
    ...
    /** 64-bit signed integer. */
    INT64 {
```

And on `from`:

```java
    /**
     * Maps a dtype name to its constant.
     *
     * @param name one of {@code float32}, {@code float64}, {@code int32}, {@code int64}, with or
     *     without a {@code torch.} prefix
     * @return the matching constant
     * @throws IllegalArgumentException if the name is not recognised
     */
    public static DType from(String name) {
```

- [ ] **Step 4: Fix `EtNative.java` (6 warnings — lines 29, 42, 44, 47, 50)**

Line 29 is `loadModule`, 42/44 are the `nativeLog` sink, 47 is `setIntraOpThreads`, 50 is `intraOpThreads`. Read each declaration and add javadoc with a `@param` for every parameter and a `@return` for every non-void method. For example:

```java
    /**
     * Sets the process-global intra-op (XNNPACK) thread pool size.
     *
     * @param n requested thread count; must be at least 1
     * @return the count in effect after the attempt, which may differ from {@code n}
     */
    public static native int setIntraOpThreads(int n);

    /** @return the current intra-op pool size as reported by the native pool */
    public static native int intraOpThreads();
```

- [ ] **Step 5: Fix `EtMethodMeta.java` (2 warnings — lines 5, 26)**

Line 5 is the `numInputs` field, line 26 the constructor:

```java
    /** Number of inputs the {@code forward} method declares. */
    public final int numInputs;
```

```java
    /**
     * @param numInputs number of declared inputs
     * @param inputScalarTypes per-input ScalarType code, {@code -1} for a non-tensor input
     * @param inputMemoryPlanned per-input memory-planned flag
     * @param plannedArenaBytes ExecuTorch's planned activation arena in bytes
     */
    public EtMethodMeta(
```

- [ ] **Step 6: Fix `EtDataTypes.java` (2 warnings — lines 11, 27), `EtEngine.java` (2 — lines 14, 153), `LibUtils.java` (1 — line 28), `MapTranslator.java` (3 — lines 25, 33, 37)**

`EtDataTypes` needs `@param dataType` and `@param scalarType` on its two conversion methods. `EtEngine.java:14` is `ENGINE_NAME` (add `/** The DJL engine name this plugin registers: {@value}. */`) and `:153` needs a `@return`. `LibUtils.java:28` is `loadLibrary()`. `MapTranslator` needs javadoc on its constructor and the two static factories, each with `@param` and `@return`.

Read each declaration before writing, and describe what it does rather than restating its name.

- [ ] **Step 7: Fix the example module (7 warnings)**

`MobilenetExample.java:18` is `main` — add `@param args` and `@throws`. `ModelArtifacts.java:12` needs `@return`, `:17` needs `@param name` and `@return`. `Variant.java:16/17/18` are the three enum constants:

```java
public enum Variant {
    /** ExecuTorch engine with the PyTorch-backed image translator. */
    ET_HYBRID("ExecuTorch", MobilenetTranslator::new),

    /** LibTorch/PyTorch engine baseline, same translator. */
    PYTORCH("PyTorch", MobilenetTranslator::new),

    /** ExecuTorch engine with a pure-Java translator, so the arm never loads LibTorch. */
    ET_NATIVE("ExecuTorch", PlainJavaMobilenetTranslator::new);
```

- [ ] **Step 8: Verify zero warnings**

```bash
./gradlew :javadoc :example:javadoc --rerun-tasks 2>&1 | grep -cE "^/home.*warning"
```

Expected: `0`. If the count is non-zero, the remaining lines name the exact file and line.

- [ ] **Step 9: Commit**

```bash
git add src/main/java example/src/main/java
git commit -m "docs: complete javadoc on the public API"
```

---

### Task 7: Comment the native layer

`native/jni/executorch_djl_jni.cpp` is 308 lines at 6% comment density and holds the subtlest code in the repository. `native/core/et_runtime.h` at 40% sets the standard to match.

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp`
- Modify: `native/jni/et_logging.cpp`
- Modify: `native/jni/array_size_limits.h`, `native/jni/et_log_level.h`, `native/core/staging.h`, `native/core/et_probes.h`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks; `docs/native-architecture.md` (Task 8) links to these files.

- [ ] **Step 1: Document the cached JNI handles and the signature hazard**

At the block of `static jclass` / `static jmethodID` globals near the top of `executorch_djl_jni.cpp`, add:

```cpp
// Class references, field IDs and method IDs cached once in JNI_OnLoad. Lookups are relatively
// expensive and FindClass is unsafe with an exception pending, so nothing here is resolved per
// call. The jclass values are global refs (see cacheGlobalClass) because local refs do not
// survive the return from JNI_OnLoad.
//
// DANGER: the method-ID signature strings below are hardcoded and are NOT checked against the
// Java side by any compiler. Change a Java constructor or method without updating its signature
// here and GetMethodID returns null -- a failure at class-init RUNTIME, not at build time. Linux
// masks it because both sides are always rebuilt together; it typically surfaces first on
// Windows. Treat the Java signature and the literal here as one edit. This is not hypothetical:
// adding EtMethodMeta.plannedArenaBytes required changing "(I[I[Z)V" to "(I[I[ZJ)V".
```

- [ ] **Step 2: Document the exception-translation helpers**

Above `throwJava` and `throwIllegalArgument`, extend the existing comments to state the contract explicitly:

```cpp
// Exception translation. Both helpers only SCHEDULE a Java exception -- they do not return
// control to the JVM, so every caller must return immediately afterwards. Continuing to call JNI
// functions with an exception pending is undefined behaviour.
//
// The two differ deliberately in how they obtain their class. throwJava is called from catch
// blocks where an exception may already be pending, and FindClass would then itself fail and
// return null -- hence the cached global ref. throwIllegalArgument is called from argument
// checks before anything can have thrown, so a per-call FindClass is safe there, and it is
// null-checked only for the case where a prior JNI call left something pending.
```

- [ ] **Step 3: Document handle lifetime at the entry points**

Above the first function that takes a `jlong handle`, add:

```cpp
// Handle convention: every jlong `handle` below is a reinterpret_cast of an EtRuntime* owned by
// the Java side. loadModule allocates it and returns ownership to Java; destroy frees it. The
// native layer holds no registry and does no validation -- a handle that has been destroyed, or
// was never returned by loadModule, is a use-after-free or a wild pointer. EtSymbolBlock enforces
// this on the Java side by zeroing its handle field under a monitor that also excludes the stats
// path; see its close() and toStats().
```

- [ ] **Step 4: Document the input-buffer liveness requirement in `forward`**

The existing comment about direct ByteBuffers staying live is correct but understates why. Extend it:

```cpp
  // The direct ByteBuffers reached through jinputs must stay live for the whole call: GetDirectBufferAddress
  // hands back a raw pointer into the JVM's off-heap memory, and ExecuTorch either borrows it
  // (unplanned inputs, copied into our staging slot first) or memcpy's from it (memory-planned
  // inputs, the export default). Holding the jobjectArray keeps its elements reachable, which is
  // what makes those addresses valid through rt->forward().
```

- [ ] **Step 5: Document the marshalling loop's local references**

Above the loop that calls `GetObjectArrayElement` per input:

```cpp
  // Each GetObjectArrayElement / GetObjectField below creates a LOCAL reference. The JNI spec
  // guarantees only 16 free local slots, though real JVMs provide far more; a model with many
  // inputs would otherwise risk exhausting the frame. The references are left to be reclaimed
  // when this native frame returns, which is correct for a bounded input count -- if a model with
  // hundreds of inputs ever appears, add explicit DeleteLocalRef calls or an EnsureLocalCapacity.
```

- [ ] **Step 6: Document `et_logging.cpp` (8%)**

Add a file-level comment stating the PAL sink contract:

```cpp
// ExecuTorch PAL logging bridge. ExecuTorch calls et_pal_emit_log_message from arbitrary native
// threads, including ones the JVM has never seen, so this file must attach to the JVM before it
// can call back into Java, and must never assume a JNIEnv is already available.
//
// Level codes crossing to Java are ours, not ExecuTorch's: 0=debug 1=info 2=warn 3=error, mapped
// in et_log_level.h. Keep that mapping in sync with EtNative.nativeLog on the Java side.
//
// A log call must never fail an inference: every failure path here degrades to dropping the
// message rather than propagating.
```

- [ ] **Step 7: Give each small header a why-it-exists sentence**

`native/jni/array_size_limits.h`:

```cpp
// Guards the jsize boundary. JNI array lengths are jsize (int32), so an ExecuTorch output larger
// than INT32_MAX bytes cannot be represented as a Java array at all. Rejecting it explicitly
// turns a silent truncation into a clean exception.
```

`native/jni/et_log_level.h`:

```cpp
// The single mapping between ExecuTorch's PAL log level characters and the integer codes we hand
// to EtNative.nativeLog. Isolated in its own header so the JNI bridge and the Catch2 units test
// the same table rather than two copies of it.
```

`native/core/staging.h`: extend the existing `StagingSlot` comment with the padding rationale:

```cpp
// Why the padding and the 64-byte alignment: XNNPACK documents an over-read allowance past the
// end of an input buffer (XNN_EXTRA_BYTES), so a slot sized exactly to the tensor would be read
// past. 64-byte alignment matches a cache line and is what aligned_alloc is asked for.
```

`native/core/et_probes.h`: extend the file header:

```cpp
// USDT/DTrace probes for the staging path, plus an in-process handler used by the unit tests.
// The probes compile to no-ops off Linux/GCC. They exist because staging growth is invisible from
// Java -- it happens entirely inside the native allocator -- and a grow on the hot path means a
// slot was under-sized at load, which is a bug worth catching in a bpftrace one-liner.
```

- [ ] **Step 8: Verify the native build still compiles**

Run: `./native/build.sh`

Expected: `BUILD SUCCESSFUL`, shim staged. Comments cannot break a build, but a stray unterminated block comment can — this catches that.

Then restore the clangd database, which `build.sh` reset:

```bash
./native/gen_clangd_db.sh
```

- [ ] **Step 9: Verify the density improved**

```bash
for f in native/jni/executorch_djl_jni.cpp native/jni/et_logging.cpp; do
  tot=$(grep -c "" "$f"); cmt=$(grep -cE "^\s*(//|/\*|\*)" "$f")
  printf "%-38s %d%%\n" "$f" $((cmt*100/tot))
done
```

Expected: both above 20%. This is a sanity signal, not a target to game — a file padded with restatements of its own code fails the intent even at 40%.

- [ ] **Step 10: Commit**

```bash
git add native/
git commit -m "docs: comment the JNI boundary and the small native headers"
```

---

### Task 8: docs/native-architecture.md

**Files:**
- Create: `docs/native-architecture.md`

**Interfaces:**
- Consumes: the `docs/` layout (Task 2); links to `building.md` (Task 5).
- Produces: the document `docs/README.md` already links.

- [ ] **Step 1: Write the document**

Create `docs/native-architecture.md` covering, in this order:

1. **The three layers.** `native/core/` is a JNIEnv-free C++ core (`measly::et::EtRuntime`) wrapping ExecuTorch's `Module`. `native/jni/` is the JNI shim plus the PAL logging bridge, and is the only part that knows a JVM exists. `native/harness/` and `native/test/` are benchmark, sanitiser and Catch2 binaries.

2. **Why the core is JNIEnv-free.** It is deliberate, not incidental: the Catch2 units, the leak harness and the timing harness all link the core directly, so a QA or bench configure needs no JDK and no `JAVA_HOME`. `native/CMakeLists.txt` enforces the split by building the shim only when `ET_BUILD_QA` and `ET_BUILD_BENCH` are both off. State the consequence for editors: no single CMake configure covers both `jni/` and `test/`, which is why `native/gen_clangd_db.sh` runs two and merges them.

3. **Ownership and data flow.** Inputs are borrowed host pointers; outputs are a single copy into a heap `byte[]`. Then the correction that matters: **the input borrow is not zero-copy end to end.** ExecuTorch's `Method::set_input` copies into its own arena whenever `is_memory_planned()` is true — the export default, and true for every `.pte` in this repository — and honours the borrow only for models exported with `alloc_graph_input=False`. Link to `executorch-host-buffer-contract-brief.md`.

4. **Staging slots.** One per input, allocated at construction from the `.pte`'s declared bound, grow-only so the steady state is allocation-free, 64-byte aligned, padded by `kStagingPadding` (128 bytes) for XNNPACK's documented over-read allowance. Planned inputs are never staged, so their slots stay at capacity 0 — which is why `EtEngineStats` reports 0 staging bytes for most real models and why that 0 is meaningful rather than missing.

5. **Process-global state, and how the two kinds differ.** The intra-op (XNNPACK) thread pool is a process singleton, write-once, applied and sealed at the first model load; a late reset is a use-after-free rather than merely a race, because XNNPACK captures the `pthreadpool_t` when it creates a runtime. Workspace sharing is the opposite: resolved per delegate at load, so modes compose across models and load order is irrelevant.

6. **Probes.** The USDT probes in `et_probes.h`, what each fires on, and a `bpftrace` invocation to observe them.

7. **Where the runtime comes from.** One paragraph: not built here, downloaded as a hash-pinned attested tarball, pin in `native/cmake/EtRuntimePin.cmake`. Link to `building.md` rather than repeating it.

- [ ] **Step 2: Check links**

Run: `./tools/scripts/check_doc_links.sh docs/native-architecture.md docs/README.md`

Expected: passes.

- [ ] **Step 3: Commit**

```bash
git add docs/native-architecture.md
git commit -m "docs: describe the native layer's architecture"
```

---

### Task 9: CLAUDE.md and the full verification sweep

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything.
- Produces: a green tree.

- [ ] **Step 1: Update CLAUDE.md for the new layout**

CLAUDE.md cites `docs/` paths directly. Update the "Design docs live in..." line at the end of the conventions list to describe the new structure, and verify every other `docs/` path it mentions still resolves:

```bash
grep -oE 'docs/[A-Za-z0-9_/.-]+\.md' CLAUDE.md | sort -u | while read -r p; do
  [ -e "$p" ] || echo "MISSING: $p"
done
```

Expected: no `MISSING` output.

- [ ] **Step 2: Whole-tree link check**

Run: `./tools/scripts/check_doc_links.sh`

Expected: passes over every tracked `.md`. This is the first run where `building.md` and `native-architecture.md` both exist, so the `docs/README.md` index rows now resolve.

- [ ] **Step 3: Confirm no references to pre-move paths survive**

```bash
grep -rIn "docs/handover-to-engine\|docs/handover-windows-static\|docs/host-buffer-contract-wip\|docs/panama-research-sketch\|docs/iree-lessons-learned" \
  --exclude-dir=.git --exclude-dir=build --exclude-dir=build-clangd --exclude-dir=_deps . \
  | grep -v "docs/research/"
```

Expected: no output.

- [ ] **Step 4: Zero javadoc warnings**

```bash
./gradlew :javadoc :example:javadoc --rerun-tasks 2>&1 | grep -cE "^/home.*warning"
```

Expected: `0`.

- [ ] **Step 5: No emoji in the README**

```bash
grep -nP "[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}\x{2B00}-\x{2BFF}\x{FE0F}]" README.md && \
  echo "EMOJI FOUND" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 6: The quickstart compiles and runs**

```bash
./gradlew :example:compileJava
./gradlew :example:runQuickStart
```

Expected: prints `2 + 3 = 5.0`.

- [ ] **Step 7: Full build**

```bash
./gradlew build
```

Expected: `BUILD SUCCESSFUL`, including `checkDocLinks` via `check`. Nothing in this plan touches behaviour; a test failure here is a genuine finding, not a flake to retry.

- [ ] **Step 8: Commit and push**

```bash
git add CLAUDE.md
git commit -m "docs: point CLAUDE.md at the new documentation layout"
git push -u origin docs/release-readiness
```

- [ ] **Step 9: Confirm CI**

```bash
gh pr create --fill --base main
gh pr checks --watch
```

Expected: all legs green. `checkDocLinks` runs on the Linux legs only, which is sufficient — the check is platform-independent.

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: document map and triage rule → Task 2; README structure → Task 4; `building.md` and the `.clangd` repoint → Task 5; the compiled quickstart → Task 3; javadoc → Task 6; native comments → Task 7; `native-architecture.md` → Task 8; untracked artifacts → Task 2 Step 6 (the `.gitignore` rule for `.settings/` already landed with the spec commit); verification → Task 1 for the tool and Task 9 for the sweep.

**Ordering is deliberate.** The link checker is Task 1 because Tasks 2, 4, 5 and 8 all use it, and because it must establish a clean baseline *before* eight files move — otherwise pre-existing breakage is indistinguishable from breakage this plan caused. `QuickStart.java` precedes the README because the README quotes it. `docs/README.md` is written in Task 2 but links two files that do not exist until Tasks 5 and 8, so the whole-tree link check is deferred to Task 9; the intermediate tasks check explicit files instead.

**Naming consistency.** `tools/scripts/check_doc_links.sh` and the Gradle task `checkDocLinks` are used identically throughout. `docs/research/` is the destination everywhere. `org.measly.example.QuickStart` and `:example:runQuickStart` match between Tasks 3, 4 and 9.

**One judgement call left to the implementer, deliberately.** Task 2 Step 3 asks for a date in each status header, taken from the file's content or `git log`. That cannot be pre-filled for the four untracked files, which have no commit history to read.

**The Task 1 script is verified, not drafted.** It was run against all eight test cases and against the real tree before this plan was committed. The first version produced two false positives — an ASan stack frame `operator new[](unsigned long)` inside a fenced block in `host-buffer-contract-wip.md`, and the literal `](...)` inside backticks in this cycle's own design spec — which is why code stripping is in the script and both cases are regression fixtures in the test. The clean baseline is 41 files.
