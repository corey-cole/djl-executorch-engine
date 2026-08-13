# Gate C — `-Xcheck:jni` Implementation Plan (PR 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every JVM test task under `-Xcheck:jni`, the JVM's JNI-contract checker, and fix whatever it reports.

**Architecture:** One `jvmArgs` line on the `tasks.withType<Test>().configureEach` block that already exists in `build.gradle.kts`, plus two test classes that prove the flag actually reached all nine test tasks — an untagged base class for `tasks.test` and a subclass carrying all eight excluded tags for the tag-filtered tasks.

**Tech Stack:** Gradle (Kotlin DSL), JUnit 5, JDK 17.

**Spec:** `docs/superpowers/specs/2026-08-13-ubsan-and-jni-checking-design.md` (§1, §3, §4)
**Source material:** `docs/research/ubsan-jni-checking-port-handover.md` §3

## Global Constraints

- **The flag goes on the `Test` umbrella, never on `tasks.test`.** `tasks.test` excludes eight tags, and `oomTest` — the task driving the allocation-failure paths this defect class lives on — is one of the excluded. Attaching to `tasks.test` alone would never visit the code the flag is for.
- **Assert the checker is *active*, never that it fires.** The null-check branches need heap exhaustion mid-loop and are not deterministically reachable, so a revert-the-fix probe is not a reproducible acceptance criterion.
- **The eight excluded tags, exactly:** `leak`, `oom`, `intraop`, `jmx-disabled`, `stats-degraded`, `stress`, `stress-sweep`, `stress-baseline`.
- **Findings get fixed in this PR.** A gate that cannot be turned on green is not done. If `-Xcheck:jni` reports a warning or aborts the VM, fix the shim — never weaken the gate.
- **Resource containment is mandatory for `oomTest`.** It exhausts a heap deliberately, and a runaway run on this project has triggered host-wide OOM kills. Always `./gradlew --stop` first, then wrap in `systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900` with `--no-daemon` inside — a pre-existing daemon lives in whatever cgroup it started in, so forked test JVMs would otherwise escape the scope.
- **No native rebuild is needed.** `src/main/resources/native/linux-x86_64/libexecutorch_djl.so` is already staged, and this gate runs against the plain shipping library.

---

### Task 1: Attach the flag and prove it on `tasks.test`

**Files:**
- Create: `src/test/java/org/measly/executorch/JniCheckFlagTest.java`
- Modify: `build.gradle.kts:44-49` (the `tasks.withType<Test>().configureEach` block)

**Interfaces:**
- Produces: `JniCheckFlagTest`, a package-private class with one `@Test` method `jvmRunsWithXcheckJni()`. Task 2 subclasses it, so it must **not** be `final` and the method must be inherited (not `private`).

- [ ] **Step 1: Write the failing test**

Create `src/test/java/org/measly/executorch/JniCheckFlagTest.java`. It sits at the package root beside `ClasspathTest`, not under `jni/`: it asserts a property of the build configuration, not the behaviour of a class in that package.

```java
package org.measly.executorch;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.management.ManagementFactory;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Asserts the JVM under test runs with {@code -Xcheck:jni}, the JNI-contract checker: it catches
 * JNI calls made with an exception already pending, and null array arguments to the
 * {@code Set*ArrayRegion} family.
 *
 * <p>The flag is attached to the {@code Test} task umbrella in {@code build.gradle.kts}, so this
 * assertion must hold for every test task rather than {@code test} alone. {@link
 * JniCheckFlagTaggedTest} inherits it into the eight tag-filtered tasks.
 *
 * <p>This asserts the checker is <em>active</em>, not that it fires. The null-check branches need
 * heap exhaustion mid-loop to reach, so a fire-on-demand probe would not be reproducible and cannot
 * serve as the gate.
 */
class JniCheckFlagTest {

    @Test
    void jvmRunsWithXcheckJni() {
        List<String> args = ManagementFactory.getRuntimeMXBean().getInputArguments();
        assertTrue(
                args.contains("-Xcheck:jni"),
                "test JVM must run with -Xcheck:jni; actual JVM arguments: " + args);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

```bash
./gradlew test --tests 'org.measly.executorch.JniCheckFlagTest'
```

Expected: FAIL — `test JVM must run with -Xcheck:jni; actual JVM arguments: [-XX:+HeapDumpOnOutOfMemoryError, ...]`.

- [ ] **Step 3: Attach the flag**

In `build.gradle.kts`, add `jvmArgs` to the existing `tasks.withType<Test>().configureEach` block (the one already declaring the `executorchLibraryPath` input), leaving that input property untouched:

```kotlin
tasks.withType<Test>().configureEach {
    inputs.property(
        "executorchLibraryPath",
        providers.environmentVariable("EXECUTORCH_LIBRARY_PATH").orElse("")
    )
    // -Xcheck:jni is the JVM's JNI-contract checker: JNI calls made with a pending exception, null
    // array arguments. It is on the umbrella rather than on tasks.test because tasks.test excludes
    // eight tags, and oomTest -- excluded -- is the task that drives the allocation-failure paths
    // this class of defect lives on. Costs nothing: it runs against the plain shipping library.
    jvmArgs("-Xcheck:jni")
}
```

- [ ] **Step 4: Run it to verify it passes**

```bash
./gradlew test --tests 'org.measly.executorch.JniCheckFlagTest'
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add build.gradle.kts src/test/java/org/measly/executorch/JniCheckFlagTest.java
git commit -m "test: run every test task under -Xcheck:jni"
```

---

### Task 2: Prove attachment across the eight tag-filtered tasks

**Files:**
- Create: `src/test/java/org/measly/executorch/JniCheckFlagTaggedTest.java`

**Interfaces:**
- Consumes: `JniCheckFlagTest` from Task 1 — inherits its `jvmRunsWithXcheckJni()` method, adds no method of its own.

- [ ] **Step 1: Write the tagged subclass**

No single class can run under all nine tasks: an untagged class is skipped by every tag-filtered task, and a tagged class is excluded from `test`. This subclass is the only thing proving the umbrella reached `oomTest`.

```java
package org.measly.executorch;

import org.junit.jupiter.api.Tag;

/**
 * Carries {@link JniCheckFlagTest}'s inherited assertion into the tag-filtered test tasks.
 *
 * <p>{@code tasks.test} excludes all eight of these tags and each filtered task includes exactly
 * one, so no single class can run under every task. This subclass is how the umbrella attachment
 * gets proven where it matters most — above all in {@code oomTest}, which drives the
 * allocation-failure paths {@code -Xcheck:jni} exists to police.
 */
@Tag("leak")
@Tag("oom")
@Tag("intraop")
@Tag("jmx-disabled")
@Tag("stats-degraded")
@Tag("stress")
@Tag("stress-sweep")
@Tag("stress-baseline")
class JniCheckFlagTaggedTest extends JniCheckFlagTest {}
```

- [ ] **Step 2: Run the six cheap tag-filtered tasks**

Filtering to this one class keeps each task from running its real (and in two cases very expensive) workload — the point here is attachment, not coverage.

```bash
./gradlew leakTest intraOpTest jmxDisabledTest statsDegradedTest \
  --tests '*JniCheckFlagTaggedTest'
```

Expected: BUILD SUCCESSFUL, with the test executing in each task. If a task reports "no tests found for given includes", its tag is missing from the subclass — fix the annotation, not the task.

- [ ] **Step 3: Run the three stress-tagged tasks, filtered**

These saturate every core when run for real, which is why CLAUDE.md keeps them out of CI. Filtered to one trivial assertion they are cheap, and attachment is all that is being proven.

```bash
./gradlew stressGate stressSweepCore stressSweepBaseline \
  --tests '*JniCheckFlagTaggedTest'
```

Expected: BUILD SUCCESSFUL in all three.

- [ ] **Step 4: Run `oomTest` under containment**

This is the task the whole design turns on, and the one that must not run unconstrained.

```bash
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 \
  ./gradlew --no-daemon oomTest --tests '*JniCheckFlagTaggedTest'
```

Expected: BUILD SUCCESSFUL. If this fails while the other seven pass, the flag is not reaching `oomTest` and the umbrella attachment is wrong — that is exactly the failure this task exists to catch.

- [ ] **Step 5: Commit**

```bash
git add src/test/java/org/measly/executorch/JniCheckFlagTaggedTest.java
git commit -m "test: prove -Xcheck:jni reaches all eight tag-filtered test tasks"
```

---

### Task 3: Run the real suites under the gate and fix what it reports

**Files:**
- Modify: `native/jni/executorch_djl_jni.cpp` (only if the checker reports something)

**Interfaces:** none. This task changes native code only in response to a diagnostic.

**Expect this task to find something.** The handover is explicit: the flag audits every JNI call in the shim on every test run. A finding presents as either a `WARNING in native method:` line on stderr or a hard VM abort — **not** as an ordinary assertion failure. If either appears, fix the shim; never weaken the gate.

- [ ] **Step 1: Run the main suite unfiltered**

```bash
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 \
  ./gradlew --no-daemon test
```

Expected: BUILD SUCCESSFUL with no `WARNING in native method:` on stderr.

- [ ] **Step 2: Run the allocation-failure and leak suites unfiltered**

These drive the paths where unchecked `env->New*` results and null array arguments actually bite.

```bash
./gradlew --stop
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 \
  ./gradlew --no-daemon leakTest oomTest
```

Expected: BUILD SUCCESSFUL, no warnings.

- [ ] **Step 3: Triage anything reported**

For each `WARNING in native method:` line, record the JNI function named and the shim call site. The two classes to expect, per the spec:

- a JNI call made while an exception is already pending — the fix is an `env->ExceptionCheck()` early-return after the call that raised it;
- a null array or object passed to a JNI function — the fix is a null check on the result of the `New*` call that produced it, before it is used.

Fix the shim at the call site. Do **not** add `-Xcheck:jni` suppressions and do not narrow which tasks carry the flag.

- [ ] **Step 4: Re-run to confirm clean**

Repeat Steps 1 and 2. Expected: BUILD SUCCESSFUL, no `WARNING in native method:` lines.

- [ ] **Step 5: Commit any fixes**

Skip this step entirely if Steps 1-2 were clean the first time.

```bash
git add native/jni/executorch_djl_jni.cpp
git commit -m "fix: <the specific JNI contract violation -Xcheck:jni reported>"
```

If the shim changed, the staged `.so` is now stale. Rebuild it before Task 4:

```bash
./native/local_build_wrapper.sh
```

---

### Task 4: Document the gate and verify the whole PR

**Files:**
- Modify: `CLAUDE.md` (the "Build & test" section), `docs/building.md` (the test-suite section)

**Interfaces:** none.

- [ ] **Step 1: Document it**

Add to both docs, at the point each describes running the test suites: every `Test` task runs under `-Xcheck:jni`; it is attached to the task umbrella in `build.gradle.kts` rather than to `tasks.test`, because `tasks.test` excludes eight tags including `oom`; a violation surfaces as a `WARNING in native method:` line or a VM abort rather than a test failure; and `JniCheckFlagTest` / `JniCheckFlagTaggedTest` exist to prove the flag is attached, so deleting either silently removes the proof.

Do not describe what the build did before this change.

- [ ] **Step 2: Confirm the shipping path is unaffected**

```bash
./gradlew clean build
```

Expected: BUILD SUCCESSFUL. The flag is a test-JVM argument only; nothing instrumented or otherwise altered reaches the published artifact.

- [ ] **Step 3: Run the host-fast shell suites**

```bash
bash native/tests/ci_workflow.sh && bash native/tests/build_config.sh && bash native/tests/docs_present.sh
```

Expected: three `PASS:` lines. None of them assert on this change; this confirms it broke nothing.

- [ ] **Step 4: Commit and open the PR**

```bash
git add CLAUDE.md docs/building.md
git commit -m "docs: document the -Xcheck:jni gate"
```

Push the branch and open the PR. CI needs no change: `build-java-package` already runs `test`, `leakTest` and `oomTest`, so all three pick the flag up from the umbrella automatically.

---

## Known gap

`-Xcheck:jni` polices the JNI contract, not native memory safety. It does not catch
`EtSymbolBlock.forwardInternal`'s missing handle check (predict-after-close), which is a
use-after-free of a native handle — that bug stays open and is out of scope here, as is UBSan
coverage of the shim, which is PR 3.
