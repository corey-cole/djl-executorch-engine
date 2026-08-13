# UBSan and JNI-checking QA gates

**Status:** approved design, not yet implemented
**Source material:** `docs/research/ubsan-jni-checking-port-handover.md` — the port handover written
2026-08-12 from `djl-iree-engine` (`15d3c74`..`1dc92bf`), where all three gates are live and green.

This spec does **not** restate the handover. It records what is different *here*: the claims that
were verified against this tree, the decisions the differences force, and the port order. Read the
handover for the mechanics and the reasoning behind each gate.

## 1. Three gates

Two classes of defect need two tools, and a third covers a file the sanitizers structurally cannot
reach.

- **Gate C — `-Xcheck:jni` on every JVM test task.** The JVM's own JNI-contract checker. Catches
  JNI-*specification* UB: calling a JNI function with an exception pending, passing a null array to
  `Set*ArrayRegion`. Costs nothing and runs against the plain shipping library.
- **Gate A — UBSan over the native QA tree.** `-fsanitize=undefined` composed onto the existing ASan
  build. No new build tree.
- **Gate B — UBSan over the JNI shim, exercised by the JVM suite.** The only configuration that
  instruments `native/jni/executorch_djl_jni.cpp`: `native/CMakeLists.txt:139` skips the shim under
  `ET_BUILD_QA`, so ASan has never reached the file where JNI bugs actually live.

**Gate C is not a lesser version of the UBSan gates.** The handover is explicit that UBSan would run
clean over all three of the bugs that motivated this work — an unchecked `env->New*` result is a
well-formed indirect call with a null *argument*, and narrowing to a signed type is defined wrapping
in C++20. UBSan earns its place on a different exposure: `alignment` over host buffers whose
alignment the JVM does not guarantee, plus `null`, `bounds`, `shift`, `return` and the float checks.

## 2. Verified against this tree

The handover's §9 says to re-derive seven things rather than port them. Result:

| Claim | Here |
| --- | --- |
| Library-path override exists, declared a `Test` input | **Yes.** `build.gradle.kts:44-48` declares `EXECUTORCH_LIBRARY_PATH` as an input on the `Test` umbrella. Gate B's no-staging property holds. |
| Scripts install at run time (handover §5) | **No — already cleared** by #40 (`da21af5`). Scripts assert and never install; `pip install ninja` and the `dnf` ASan install are gone. |
| Sanitizer runtime NEVRA per arch | **Baked and self-asserted.** `gcc-toolset-14-libubsan-devel-14.2.1-11.el8_10` is installed in the shared image and asserted by its own build. No image work, and no consumer-side re-check. |
| `tasks.test` tag structure | **Excludes eight tags**: `leak`, `oom`, `intraop`, `jmx-disabled`, `stats-degraded`, `stress`, `stress-sweep`, `stress-baseline`. `oomTest` exists at `-Xmx128m`. |
| Ownership-handback helper | **Absent.** `native/build.sh` carries an inline `cleanup()` trap. §5 below. |
| Artifact pattern merged into resources | `executorch-libs-*` → `merge-multiple: true` → `src/main/resources/native/`. §7 below. |
| Container `JAVA_HOME` vs Gradle's JDK | Corretto 8 in the image; Gradle needs JDK 17. The two-phase split applies. |

Two of these are load-bearing differences from the sibling and are treated in their own sections
(§5, §7). The rest are favourable: the handover's single most expensive trap — run-time installs
silently defeating image pinning — does not exist here, because #40 removed it for unrelated reasons.

## 3. Port order and PR shape

Three PRs, in this order. Each is independently useful and independently revertable.

1. **PR 1 — Gate C.** One line plus two test classes. No native build.
2. **PR 2 — Gate A.** A CMake option plus flags in `build_qa.sh`.
3. **PR 3 — Gate B**, preceded in the same PR by the `container_env.sh` extraction (§5), plus CI
   wiring (§7).

If Gate C reports findings, they are fixed **in PR 1**. A gate that cannot be turned on green is not
done, so the gate lands enforcing rather than pending.

## 4. Gate C — proving attachment across nine tasks

The flag attaches to the `tasks.withType<Test>().configureEach` block that already exists at
`build.gradle.kts:44`, not to `tasks.test`. `tasks.test` excludes eight tags, and `oomTest` — the
task that drives the allocation-failure paths where this defect class lives — is among the excluded.

Proving attachment is harder here than in the sibling, which had four test tasks to our nine. No
single class can run under all of them: an untagged class is skipped by every tag-filtered task, and
a tagged class is excluded from `test`. So:

- An **untagged base class** asserting `ManagementFactory.getRuntimeMXBean().getInputArguments()`
  contains `-Xcheck:jni`. This covers `tasks.test`.
- One **subclass carrying all eight tags**, inheriting the `@Test` method. Each tag-filtered task's
  `includeTags` matches one of its tags, so it runs in all eight.

Assert the checker is *active*, never that it fires: the null-check branches need heap exhaustion
mid-loop and are not deterministically reachable, so a revert-the-fix probe is not a reproducible
acceptance criterion.

## 5. `native/container_env.sh` — a prerequisite, not a nicety

The sibling extracted its chown-on-exit trap into a sourced helper. We never did; `native/build.sh`
has it inline, and `build_qa.sh` has none, which is why `native/asan/` comes back root-owned today
(CLAUDE.md documents the manual `chown` as the workaround).

Gate B makes this urgent rather than cosmetic. Its build phase runs as root in the container and its
next run starts with `rm -rf` on that tree — the handover reports this exact sequence failing with a
bare `Permission denied` that names neither the container nor the cause, discovered *after* the code
had passed review once.

So PR 3 extracts the trap into `native/container_env.sh` (register paths, chown on exit, no-op when
`HOST_UID` is unset) and moves all three consumers onto it: `build.sh`, `ubsan_gate.sh`, and
`build_qa.sh`. `build_qa.sh` is included rather than deferred — it is one call once the helper
exists, and it retires the manual `sudo chown -R` on `native/asan/` that CLAUDE.md currently
documents as the workaround. `bench.sh` and `build_variants.sh` keep their current behaviour; they
are out of this PR's path.

## 6. Gates A and B — mechanics

Both are Linux-only; MSVC has no UndefinedBehaviorSanitizer, and every Windows path stays untouched.

**Gate A.** An `ET_UBSAN` option in `native/CMakeLists.txt` with checks
`undefined,float-cast-overflow,float-divide-by-zero`, `-fno-sanitize=vptr`,
`-fno-sanitize-recover=undefined`, composed onto the existing ASan configure in `build_qa.sh`'s Linux
branch, with `UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`. The two float checks are added
because GCC excludes both from its `undefined` umbrella; `vptr` is dropped because it needs every TU
holding a polymorphic object instrumented and would misfire against the uninstrumented prebuilt
runtime; `-fno-sanitize-recover` is what makes it a gate rather than a diagnostic stream.

GCC has no ignorelist — `-fsanitize-ignorelist` and `-fsanitize-blacklist` are clang-only and
`UBSAN_OPTIONS=suppressions=` does not suppress. To silence a diagnostic, prefer
`__attribute__((no_sanitize("undefined")))` on the specific function. Note the known gap:
`implicit-signed-integer-truncation` is clang-only, so that class stays uncovered on GCC.

**Gate B.** `native/ubsan_gate.sh`, build tree `native/ubsan`, two phases selected by
`ET_UBSAN_MODE` defaulting to `auto` — build-only when `MEASLY_DJL_PINNED_IMAGE` is set, both phases
otherwise. The shim target takes `-static-libubsan`, without which the `.so` carries a dynamic
`libubsan` dependency and `System.load` fails. The test phase must **refuse** inside the image,
naming the JDK mismatch and printing the follow-up commands, rather than letting Gradle fail
obscurely.

The instrumented library is never staged into `src/main/resources`; it is reached through
`EXECUTORCH_LIBRARY_PATH`, which `LibUtils` honours ahead of the classpath copy. Run with
`--rerun-tasks`: a cached `UP-TO-DATE` reports a pass for a run that never happened.

**Test tasks driven by Gate B: `test`, `oomTest`, `leakTest`.** These are the marshalling-heavy and
allocation-failure paths, and they are exactly what `build-java-package` already runs. The six
config-variant and stress tasks are excluded: `intraOpTest`, `jmxDisabledTest` and
`statsDegradedTest` vary Java-side configuration rather than shim behaviour, and the three stress
tasks are kept out of CI deliberately because they saturate every core.

**A UB hit presents as a JVM hard crash**, not a test failure. The finding is the `runtime error:`
line and its stack trace *above* the JVM's crash output. This belongs in the script header, or it
reads as a flake.

## 7. CI wiring

Gates A and C need no new job: Gate A rides inside `build_qa.sh`, which
`native-build-job.yml` already runs, and Gate C rides inside the existing Gradle invocations.

Gate B spans two jobs, following the seam the workflows already have:

- **Build phase** in the `linux-x86_64` container job of `native-build-job.yml`. Primary platform
  only — a second native build plus a `--rerun-tasks` suite is real CI time, and the aarch64 row
  would double it for no new defect class.
- **Test phase** in a new job modelled on `build-java-package` in `native-build.yml`: JDK 17, download
  the instrumented library, export `EXECUTORCH_LIBRARY_PATH`, run the three tasks.

**The artifact is named `executorch-ubsan-linux-x86_64`, deliberately outside `executorch-libs-*`.**
`native-build.yml:36-38` downloads that pattern with `merge-multiple: true` directly into
`src/main/resources/native/`. An instrumented library matching it would become what every ordinary
test run loads, silently. This is a correctness requirement, not tidiness.

Cost to note: a fresh CMake tree re-runs `FetchContent`, so the pinned runtime tarball is downloaded
once more per CI run than before.

## 8. Verification

Prove each gate fails when it should, rather than trusting a passing run:

- **Gates A and B:** add a temporary UB expression (`(void)(volatile_int << 99);`) to the relevant
  translation unit, confirm the `runtime error:` diagnostic *and* a nonzero exit, then revert. Never
  commit the probe. For Gate B this is the single most important check, since the gate exists only to
  cover `executorch_djl_jni.cpp`.
- **Gate C:** assert the flag reaches every test task (§4).
- **Regression:** the plain build and plain test suite still pass, confirming nothing instrumented
  leaked into the shipping path.

Resource containment is not optional. `oomTest` exhausts a heap deliberately, and a runaway run on
these projects has triggered host-wide OOM kills. Run under
`systemd-run --user --scope -p MemoryMax=4G taskset -c 0-3 timeout 900 …`, with `./gradlew --stop`
first and `--no-daemon` inside — a pre-existing daemon lives in whatever cgroup it started in, so the
forked test JVMs would otherwise escape the scope entirely. A host scope does not contain a
container either; limit the container directly with `--memory=8g --memory-swap=8g --cpuset-cpus=0-3`.

## 9. Out of scope

- **`EtSymbolBlock.forwardInternal`'s missing handle check** (predict-after-close crashes the JVM).
  It is a use-after-free of a native handle, which neither `-Xcheck:jni` nor UBSan reliably catches.
  Named here only so it is not mistaken for something these gates cover.
- **A clang-based UBSan variant**, which would bring `implicit-signed-integer-truncation` and a real
  ignorelist. A clean follow-on, not a prerequisite. The shared image ships no clang.
- **Windows**, permanently: MSVC has no UBSan.
- **TSan**, which the sibling also carries. Not requested, and this repo's threading story is already
  covered by the stress suite.
