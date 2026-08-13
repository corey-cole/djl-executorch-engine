# Porting the UBSan and JNI-checking gates to the ExecuTorch engine

Date: 2026-08-12
Source: `djl-iree-engine`, commits `15d3c74`..`1dc92bf`
Spec: `docs/superpowers/specs/2026-08-12-ubsan-and-jni-checking-design.md`
Plan: `docs/superpowers/plans/2026-08-12-ubsan-and-jni-checking.md`

Written for a fresh agent session working in the sibling DJL ExecuTorch engine repo, with
that source tree as context. It assumes the two repos share a general shape — a `native/`
tree with `core/`, `jni/`, `harness/`, `test/`, a `build.sh` / `build_qa.sh` pair, pinned
`docker/` build images, a Gradle build with tag-filtered test tasks — and it deliberately
does **not** assume any specific file or line. Where this document says "verify", verify;
do not port on faith.

The work here is complete and green: three QA gates, blocking in CI, plus the toolchain
corrections they forced.

---

## 1. What was built, and why each gate exists

Three gates, because two different classes of defect need two different tools, and the
third covers a file the existing sanitizers structurally cannot reach.

**Gate A — UBSan over the native QA tree.** `-fsanitize=undefined` composed onto the
existing ASan build. No new build tree; a CMake option plus flags.

**Gate B — UBSan over the JNI shim, exercised by the JVM suite.** The shim is skipped by
ASan and TSan builds (they stay JVM-free deliberately), so it had never been reached by any
sanitizer — despite being where every JNI bug actually lived. UBSan can reach it because it
needs no runtime preload: `-static-libubsan` folds its runtime into the `.so`, so a stock
JVM can `dlopen` it.

**Gate C — `-Xcheck:jni` over every JVM test task.** The JVM's own JNI-contract checker.

### The correction that shapes the whole design

**UBSan does not catch JNI-specification UB.** This repo's issue 16 — unchecked
`env->New*` results in the output marshalling loop, whose sibling is your **issue #11** —
is *JNI-spec* UB: calling a JNI function with an exception already pending, passing a null
`jlongArray` to `SetLongArrayRegion`. UBSan sees a well-formed indirect call through the
`JNIEnv` table with a null *argument*. No null dereference, no overflow, nothing in its net.

Issue 15 (a size truncating through a 32-bit `jint`) is not C++ UB either: narrowing
conversion to a signed type is well-defined wrapping in C++20.

**All three of the bugs that motivated this work would have run clean under UBSan.** The
tool that catches that class is `-Xcheck:jni`, which is a JVM flag, costs nothing, and runs
against the plain shipping library.

If you port only one gate, port Gate C. It is one line, it needs no new build, and it is
the one aimed at the defect class you have already filed.

UBSan still earns its place, on a different exposure: the `alignment` check over host
buffers whose alignment the JVM does not guarantee, plus `null`, `bounds`, `shift`,
`return`, `unreachable` and the float checks across the core and shim.

---

## 2. Port order

Cheapest and highest-yield first. Each step is independently useful; stop anywhere.

1. **Gate C** (`-Xcheck:jni`). One line plus two small test classes.
2. **Gate A** (UBSan on the QA tree). A CMake option plus flags.
3. **Toolchain pinning** (§5). Do this *before* Gate B if your `build_qa.sh` installs
   packages at run time — otherwise Gate A quietly defeats your image pinning.
4. **Gate B** (UBSan on the shim). New tree, new script, two phases.
5. **CI wiring**, once each is green locally.
6. **Docs**: contributor guide plus agent-facing trip-wires.

---

## 3. Gate C — the one to port first

Attach the flag to the **`Test` task umbrella**, not to `tasks.test`:

```kotlin
tasks.withType<Test>().configureEach {
    // -Xcheck:jni is the only lever that catches issue 16's defect class: JNI
    // calls made with a pending exception, and null array arguments. It is on the
    // umbrella rather than on tasks.test deliberately -- tasks.test excludes the
    // leak/oom/stress tags, and oomTest is the one task that drives the
    // allocation-failure paths those bugs lived on. Costs nothing: it runs
    // against the plain shipping library.
    jvmArgs("-Xcheck:jni")
}
```

**Verify first:** does your `tasks.test` exclude tags? Here it does
`excludeTags("leak", "oom", "stress")`, with separate `leakTest` / `oomTest` / `stressTest`
tasks each including exactly one. If yours is the same, attaching to `tasks.test` alone
would **never visit the code the flag is for** — the OOM-path task is precisely the one
excluded. Check your tag structure before copying.

### Proving the flag is actually attached

Assert the checker is *active*, not that it fires. This repo's OOM test documents that the
null-check branches "are not deterministically reachable: they need heap exhaustion
mid-loop, and the large output fails first at the already-checked
`ByteBuffer.allocateDirect`." A revert-the-fix probe is therefore not reproducible and must
not be an acceptance criterion.

```java
@Test
void jvmRunsWithXcheckJni() {
    List<String> args = ManagementFactory.getRuntimeMXBean().getInputArguments();
    assertTrue(
            args.contains("-Xcheck:jni"),
            "test JVM must run with -Xcheck:jni; actual JVM arguments: " + args);
}
```

**No single class can run under all four tasks** — an untagged class is skipped by the
tag-filtered tasks, and a tagged class is excluded from `test`. Use an untagged base class
plus a subclass carrying all three tags, inheriting the `@Test` method. That subclass is
the only thing proving the umbrella attachment reached the task where it matters most.

**Expect this step to find something.** It audits every JNI call in the shim on every test
run. If it reports a warning or aborts the VM, fix the shim — do not weaken the gate.

---

## 4. Gate A — UBSan on the QA tree

A CMake option that **composes with ASan** (unlike the ASan/TSan pair, which must stay
mutually exclusive). Verbatim from this repo, and portable as-is:

```cmake
option(<PREFIX>_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)

set(<PREFIX>_UBSAN_CHECKS "undefined,float-cast-overflow,float-divide-by-zero"
    CACHE STRING "UBSan check set passed to -fsanitize=")

if(<PREFIX>_UBSAN)
  if(WIN32)
    message(FATAL_ERROR "<PREFIX>_UBSAN is unsupported on Windows: MSVC has no UndefinedBehaviorSanitizer")
  endif()
  add_compile_options(
      -fsanitize=${<PREFIX>_UBSAN_CHECKS}
      -fno-sanitize=vptr
      -fno-sanitize-recover=undefined
      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${<PREFIX>_UBSAN_CHECKS})
endif()
```

Then pass `-D<PREFIX>_UBSAN=ON` alongside your existing ASan flag in `build_qa.sh`'s Linux
branch, and export `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`.

### The four decisions embedded above

- **`float-cast-overflow` and `float-divide-by-zero` are added** because GCC deliberately
  excludes both from its `undefined` umbrella.
- **`vptr` is dropped.** It needs every TU holding a polymorphic object instrumented and is
  the check most likely to misfire against an uninstrumented prebuilt runtime. If your RAII
  handles are non-polymorphic, it buys nothing.
- **`-fno-sanitize-recover=undefined` is what makes this a gate.** UBSan's default is
  print-and-continue: CI stays green while diagnostics scroll past.
- **Linux only.** MSVC has no UBSan. Leave every Windows path untouched.

### GCC has no ignorelist — measured, not assumed

`-fsanitize-ignorelist` and `-fsanitize-blacklist` are **both rejected as unrecognized
options** by GCC (13.3 tested); they are clang-only. The runtime fallback does not work
either: `UBSAN_OPTIONS=suppressions=<file>` with a `null:` entry still reported and still
exited nonzero.

To silence a diagnostic on GCC, in order of preference:

1. `__attribute__((no_sanitize("undefined")))` on the specific function (verified working).
2. A per-TU `set_source_files_properties(... COMPILE_OPTIONS "-fno-sanitize=<check>")`.
3. `-fno-sanitize=<check>` program-wide — last resort; it silently removes a check from the
   gate, so it needs a comment naming what was given up.

**Known gap:** the check that catches implicit truncation is
`-fsanitize=implicit-signed-integer-truncation`, which is **clang-only**. On GCC that class
stays uncovered. A clang variant later would bring both that check and a real ignorelist;
it is a clean follow-on, not a prerequisite.

---

## 5. Toolchain pinning — do this before wiring UBSan into CI

**This is the trap that cost the most time here, and it is likely present in your repo
too.** Our images pinned carefully — dated base tag, exact package NEVRA, pinned ninja,
versioned JDK download with a checksum — and then the *scripts* undid it at run time:

- The guard used a **bare package name** (`rpm -q --quiet <pkg>`), so any version satisfied
  it and the pinned NEVRA was never actually verified.
- The new sanitizer runtime was not baked into the image, so that guard failed inside the
  container and `dnf install -y -q` pulled an **unpinned** package on every run, with
  `|| true` swallowing failures.
- `build.sh` did the same with an unpinned `pip install ninja` against an image pinning an
  exact ninja version.
- The "fallback for host runs" branch was **dead code**: it used `rpm`/`dnf` on an Ubuntu
  workstation and Ubuntu runner, where `rpm` is command-not-found and `command -v dnf` is
  false. Under `set -euo pipefail` the whole block was a silent no-op that had never run.

### The rule that fixes it

**Scripts assert; they never install.** Inside the image an install defeats the pinning.
Outside it, an install is unpinned by construction and silently changes what the run
measures. Both are failures, so both fail.

1. Bake the sanitizer runtime into the image at the **same NEVRA as the existing one** — it
   must match the compiler that emitted the instrumentation, or you get confusing link
   errors. Resolve it per-architecture on real hardware; do not assume the aarch64 repo
   carries the same revision as x86_64. (Ours did, verified by querying an actual aarch64
   host, not qemu.)
2. Have the image **publish its own pins** as environment variables, so the scripts assert
   against values they cannot drift from:
   ```dockerfile
   ENV <PREFIX>_PINNED_IMAGE=1
   ENV <PREFIX>_TOOLSET_VER=<n>
   ENV <PREFIX>_TOOLSET_NEVRA=<exact-nevra>
   ENV <PREFIX>_NINJA_VERSION=<what `ninja --version` prints>
   ```
   Keep the NEVRA identical to the install line above it — the Dockerfile stays the single
   source of truth. Add image-build-time assertions (`rpm -q` the exact NEVRA) so a pin that
   stops delivering fails at image build, not three steps into a QA run.
3. In the scripts: if the marker is set, **assert** and exit nonzero with a `BROKEN IMAGE`
   message on a miss. Otherwise, **probe** distro-agnostically — compile a trivial file with
   `-fsanitize=address` and `-fsanitize=undefined` and fail with package hints if either
   cannot link. That tests the property the build depends on rather than a package-name
   proxy, and it works on any distro.
4. **Guard the companion variables** under `set -u`:
   ```bash
   : "${<PREFIX>_TOOLSET_VER:?<PREFIX>_PINNED_IMAGE is set but <PREFIX>_TOOLSET_VER is not -- rebuild the image, or unset the marker for a host run}"
   ```
   Without this, a hand-set marker or an image predating the ENV block aborts with bash's
   bare "unbound variable" — the one path through a block whose entire purpose is a legible
   message.

Two gotchas worth knowing in advance: the pinned `ninja` wheel may report a version string
unlike its package version (ours prints `1.13.0.git.kitware.jobserver-pipe-1` for
`ninja==1.13.0`), so compare against what `ninja --version` actually prints; and check
whether your `build.sh` host branch exports a container-only interpreter path (ours did,
vestigially, purely so the adjacent `pip install` resolved).

---

## 6. Gate B — UBSan on the shim, and the two-phase split

The highest-value gate and the fiddliest. It is the only configuration that instruments the
JNI shim.

### Mechanics

- Allow the shim target to build under UBSan (your existing guard almost certainly excludes
  ASan and TSan only, so UBSan may already fall through — check).
- Add `-static-libubsan` to that target's link options. **Without it the `.so` carries a
  dynamic `libubsan` dependency and `System.load` fails.** Verified: with it, the shared
  object has ~50 defined ubsan symbols and no `libubsan` in `ldd`.
- Reach the instrumented library through your **library-path environment override** — the
  variable your `LibUtils` equivalent honours ahead of the classpath copy. Nothing is staged
  into JVM resources, so the plain tree stays intact and no rebuild is needed afterwards.
  **Verify this seam exists in your repo.** If it does not, Gate B needs it added first, or
  a staging-and-restore approach instead. Also confirm the variable is declared as a `Test`
  task input, or Gradle will replay a cached pass for a run that loaded a different library.
- Run with `--rerun-tasks`. A cached `UP-TO-DATE` reports a pass for a run that never
  happened.

### The two-phase split — likely your blocker too

**Gradle probably cannot run inside your pinned build container.** Ours sets
`JAVA_HOME` to a JDK 8 (deliberately: the oldest supported `jni.h`, matching what the
Windows job binds), while the wrapper is Gradle 9.6.1 with a JDK 17 toolchain. Gradle
cannot start there at all.

**Do not fix this by adding a modern JDK to the image.** That undoes the compatibility
floor the old JDK is there for, bloats a pinned image, and creates a `JAVA_HOME` ambiguity
in the shipping build.

Split the gate instead:

- **build phase** — needs the compiler and `jni.h`. Runs in the container, which is where
  instrumentation *should* be produced: pinned toolchain, pinned runtime NEVRA.
- **test phase** — needs Gradle and a modern JDK. Runs on the host, against the `.so` the
  build phase left behind.

Select with a mode variable defaulting to `auto`: build-only when the pinned-image marker is
present, both phases otherwise. The script then knows where it is and no caller has to
remember. Make the test phase **refuse** inside the image with a message naming the JDK
mismatch and printing the two follow-up commands, rather than letting Gradle fail
obscurely. Check the JDK major version on the host path too.

### Which test tasks to run

All of them — `test leakTest oomTest stressTest` in our naming. `tasks.test` excludes the
tags, and the OOM task is the scripted reproduction of the issue-16/#11 defect class. If
your OOM task needs a compiler toolchain to build its fixture that CI does not carry, keep
it local and **say so explicitly** in the contributor docs: the reproduction is then a local
gate, not an enforced one, and nobody will notice unless it is written down.

### Two failure modes we hit

- **Root-owned outputs.** The container runs as root, so the build tree comes back
  root-owned and the *next* run's `rm -rf` dies with a bare "Permission denied" naming
  neither the container nor the cause. If your repo has an ownership-handback helper that
  `build.sh`/`build_qa.sh` source, **the new gate script must source it too.** This bit us
  after the code was written and passed review once.
- **A UB hit is a JVM hard crash**, not a test failure. The finding is the `runtime error:`
  line and its stack trace *above* the JVM's crash output. Say this in the script header, or
  it reads as a flaky test.

---

## 7. CI wiring

- **Gates A and C usually need no new job** — they ride inside the QA script and the
  existing Gradle invocations respectively.
- **Gate B spans two jobs**, following the seam your workflows almost certainly already
  have: a container job that builds native artifacts and uploads them, and a JVM job with a
  modern JDK that downloads them. Build the instrumented library in the first, upload it,
  run the JVM phase in the second.
- **Name that artifact outside the pattern your JVM job merges into the resources
  directory.** Ours downloads `iree-libs-*` with `merge-multiple: true` straight into
  `src/main/resources/native/`. An instrumented library landing there becomes what every
  ordinary test run loads — silently. This deserves an explicit check, not care.
- Scope Gate B to your primary platform only. A second native build plus a `--rerun-tasks`
  JVM suite is real CI time.
- Note the cost: a fresh CMake tree re-runs `FetchContent`, so the pinned runtime tarball is
  downloaded once more per CI run than before.

---

## 8. Verification, and how to run it safely

**Resource containment is not optional here.** A runaway test on both of these projects has
triggered host-wide OOM kills that took down unrelated processes. The OOM test exhausts a
heap deliberately.

```bash
systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 bash -c '<cmd>'
```

- **Gradle escapes this.** A pre-existing daemon lives in whatever cgroup it started in, so
  the work — including forked test JVMs — happens outside your new scope. Run `./gradlew
  --stop` first and pass `--no-daemon` inside. Confirm via `/proc/<pid>/cgroup` naming a
  `run-*.scope`.
- **A host scope does not contain a container.** With a root Docker daemon, container
  processes are children of `containerd-shim` in the system slice. Limit the container
  directly: `--memory=8g --memory-swap=8g --cpuset-cpus=0-3` (swap must equal memory, or
  Docker grants an equal amount by default and the box thrashes instead of failing fast).
- `taskset -c 0-3` caps build parallelism for free: `nproc` honours CPU affinity.
- Cold native builds need a longer timeout than 900s — `FetchContent` pulls the runtime
  tarball before compiling.

**Prove each gate fails when it should**, rather than trusting a passing run:

- Gate A / B: add a temporary UB expression (`(void)(volatile_int << 99);`) to the relevant
  translation unit, confirm the `runtime error:` diagnostic and a **nonzero** exit, then
  revert. Never commit the probe. For Gate B this is the single most important check — the
  gate exists only to cover that file.
- Gate C: assert the flag reaches every test task (§3).
- Pinning: override the NEVRA variable to a bogus value and confirm a `BROKEN IMAGE` message
  and nonzero exit, with no install attempted.
- Regression: plain build plus plain test suite still pass, confirming nothing instrumented
  leaked into the shipping path.

---

## 9. Verify, do not assume

Explicitly re-derive these in your tree rather than porting them:

1. **The library-path override exists** and is declared as a `Test` task input. Gate B's
   whole no-staging property rests on it.
2. **The test-task tag structure** — whether `tasks.test` excludes tags, and which task
   drives allocation failure.
3. **What the container's `JAVA_HOME` actually is**, and your Gradle version's JDK floor.
4. **Whether your scripts install anything at run time** (§5). Grep for `dnf`, `yum`, `apt`,
   `pip install` in `native/*.sh`.
5. **The exact sanitizer-runtime NEVRA per architecture**, on real hardware.
6. **Whether an ownership-handback helper exists**, and wire the new script into it.
7. **The artifact-name pattern** your JVM job merges into JVM resources.

## 10. Things that will be different

Your ExecuTorch tree carries directories this one does not (a no-ASan QA tree, a spike area,
build-variant scripts). Expect the runtime dependency story to differ — ours consumes a
hash-pinned prebuilt runtime tarball, and the reasoning in §4 about "an uninstrumented
prebuilt runtime produces no UBSan false positives" holds for any such arrangement, but
check whether yours is built from source, in which case you *could* instrument it and may
not need to.

One thing that will not differ: the JNI marshalling boundary is where the bugs are, and
`-Xcheck:jni` is what finds them.
