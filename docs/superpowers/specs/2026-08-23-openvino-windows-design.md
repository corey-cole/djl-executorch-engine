# OpenVINO on windows-x86_64

**Date:** 2026-08-23
**Follows:** the `v1.4.1-2` pin bump (`docs/superpowers/specs/2026-08-22-et-1.4.1-pin-bump-design.md`)
**Issue:** [#53](https://github.com/corey-cole/djl-executorch-engine/issues/53)

Sub-project **B** of the v1.4.1-2 migration. A adopted the runtime and deliberately declined the
Windows OpenVINO bundle the pin publishes, because nothing above `build.sh` could load it. This
supplies that.

## 1. What already works, and what does not

The `v1.4.1-2` runtime ships `openvino_backend.lib` in both Windows tarballs, so
`native/CMakeLists.txt`'s `if(TARGET openvino_backend)` already links the delegate on Windows and
`EtNative.backendRegistered("OpenvinoBackend")` already reports true there. The pin publishes
`openvino-runtime-2025.4.1-windows-x86_64.tar.gz`, and `native/build.sh` resolves its URL correctly
— A keyed the lookup on platform identity for exactly this reason. What blocks it is one line:
`ET_OPENVINO_SUPPORTED_PLATFORMS` does not list `windows-x86_64`, because `OpenVinoRuntime` cannot
extract what that row points at.

Three properties carry over unchanged and are worth stating so nobody re-solves them:

- **One flat directory still resolves the graph.** Linux relies on `RPATH=$ORIGIN`. The producer's
  vendored Windows port loads an absolute path with
  `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`, which searches the loaded
  DLL's own directory for its dependencies while keeping `System32` in the search order. That is the
  Windows analogue, and it means the extraction model does not change.
- **The cache and publication model already handle Windows.** `LibUtils.cacheRoot()` resolves
  `%LOCALAPPDATA%\executorch-djl\` there, and `OpenVinoRuntime.publish()` extracts to a staging
  directory and publishes by atomic rename precisely because Windows refuses to delete a loaded
  library.
- **`OPENVINO_LIB_PATH` is still the only mechanism**, and `native/jni/executorch_djl_jni.cpp`
  already calls `_putenv_s` under `_WIN32`.

What does not carry over is the bundle's *shape*, which differs in three ways at once.

| | linux-x86_64 | windows-x86_64 |
|---|---|---|
| libraries | 7 | 6 — no separate hwloc; it is folded into `tbbbind_2_5.dll` |
| naming | ABI-versioned (`libopenvino.so.2541`), except the unversioned CPU plugin | all unversioned |
| `BUILDINFO` `ov_abi` | present | **absent** — the producer emits it only where the soname symlink is needed |
| `licenses/` | 5 files, incl. `hwloc-COPYING` | 4 files |
| C API library | `libopenvino_c.so.<abi>` | `openvino_c.dll` |

Each of those lands on `OpenVinoRuntime`, whose `LIBS` list, `publish()` and `resolvedLibPath()` are
all `.so`-and-ABI shaped.

## 2. Scope

**In scope**

- The bundle declares its own contents in `MANIFEST`; `OpenVinoRuntime` reads them.
- `windows-x86_64` joins `ET_OPENVINO_SUPPORTED_PLATFORMS`, with a Windows arm in the staging step.
- The "no delegate in this build" error branch is corrected and covered by a test.
- `native/tests/openvino_bundle_staging.sh` becomes platform-parameterized.
- `openvinoTest` runs on Windows in CI.
- Docs: `CLAUDE.md`, `docs/openvino-version-bump.md`.

**Out of scope**

- `linux-aarch64`. It links the delegate and upstream publishes no bundle for it. That third state —
  runnable the moment a runtime is supplied — is deliberate and unchanged, and it keeps the
  `openvinoUnsupportedTest` leg.
- The `example/` JMH OpenVINO comparison, which stays Linux-only.
- Any OpenVINO version change. The bundle stays `2025.4.1`.

## 3. The bundle declares its own contents

`native/build.sh` writes `MANIFEST` at staging time. It gains two keys, generated from what actually
landed in `lib/`:

```
libs=openvino_c.dll openvino.dll openvino_intel_cpu_plugin.dll openvino_ir_frontend.dll tbb12.dll tbbbind_2_5.dll
c_library=openvino_c.dll
```

Space-separated is safe: no OpenVINO library filename contains a space on either platform. Because
both values are derived from the staged tree, they cannot disagree with it.

`OpenVinoRuntime` then reads rather than constructs. This retires three Windows blockers and one
pre-existing wart in a single move:

- the hardcoded seven-entry `LIBS` list;
- `resolvedLibPath()`'s `libopenvino_c.so.<abi>` construction, which has no Windows analogue at all —
  there is no `ov_abi` key to read;
- `publish()`'s ABI-suffix append;
- `publish()`'s resource-exists fallback for `libopenvino_intel_cpu_plugin.so`, which ships
  unversioned while its siblings do not. The special case disappears, because the staged name is
  simply the name.

`ov_abi` stops being read by Java entirely. That is the point: Windows works because the ABI concept
left the code path, not because a special case tolerates its absence.

**The independent cross-check does not move.** `native/tests/openvino_bundle_staging.sh` keeps its
own enumeration, derived from `BUILDINFO`, and `publish.yml` keeps gating the release on it. A
manifest generated from a truncated bundle would describe that truncation accurately and happily —
so the thing that catches truncation must not be the manifest.

## 4. `OpenVinoRuntime`

`LIBS` is deleted. `publish()` iterates `manifest.libs`; `resolvedLibPath()` returns
`extracted.resolve(manifest.c_library)`.

The "this build does not provide the delegate" branch **stays**. It is still reachable: the
`ET_INSTALL` escape hatch links a caller-supplied runtime tree, which may be built without OpenVINO,
and a future platform could ship without one. Its comment currently says "Windows today", which is
now false and is precisely how a future reader would conclude Windows has no delegate. The comment
is corrected to state the real condition, and `OpenVinoRuntimeTest` gains a case driving the branch
directly so it does not decay into an untested string.

The two error messages stay distinct. They describe situations whose remedies are opposite —
re-export the model, versus add a runtime artifact — and merging them would recreate the confusion
the split was written to prevent. After this change `linux-aarch64` is the sole shipped platform
reaching the second one.

`validateOverride`'s message names `<dir>/libopenvino_c.so.<abi>` as its example of a library file.
Where a bundle is present that example comes from `c_library` instead, so a Windows operator is not
told to point at a `.so`.

## 5. Staging and packaging

`native/build.sh`: add `windows-x86_64` to `ET_OPENVINO_SUPPORTED_PLATFORMS`, and guard the
Linux-only symlink removal rather than relying on `rm -f` to miss a file that is not there. Git-Bash
supplies `curl`, `tar`, `sha256sum` and `du`, so the rest of the staging arm is portable as written.
The `MANIFEST` write gains the two new keys on both platforms.

**Gradle needs no platform edits.** `nativePlatforms` already contains `windows-x86_64`;
`nativeJar-<platform>-openvino`'s `onlyIf` and the `openvinoVariants` filter both key on a staged
`MANIFEST`, so the Windows variant registers itself as soon as a bundle is staged. This is stated as
a prediction to verify, not an assumption to build on: a wrong answer fails at
`generateMetadataFileForMavenPublication`, which is a release-time failure, so the plan proves it
locally with a `publishToMavenLocal` against a staged Windows tree.

## 6. Tests and CI

`native/tests/openvino_bundle_staging.sh` is `linux-x86_64`-shaped in four places: the staged path,
the seven-member library list, the literal count of 7, and the ABI derivation from `BUILDINFO`. It
becomes platform-parameterized with a per-platform expected member set, still independent of
`MANIFEST`. The ABI derivation stays **Linux-only**: `ov_abi` is absent from the Windows
`BUILDINFO`, so a Windows run must not read it — the expected Windows members are literal
unversioned names, and the test asserts `ov_abi` is *absent* there rather than skipping the check.

CI runs Windows `openvinoTest` as **steps inside the existing `build-executorch-shim-windows` job**,
not as a new job. That follows the precedent already set on Linux, where `openvinoTest` runs in the
row that staged the bundle rather than in `build-java-package` — whose download glob does not match
the bundle's versioned filenames, so the bundle never reaches it. The Windows row already uploads
the whole staged tree, with a comment saying the shape exists for when Windows publishes a bundle.
Reusing the job costs no additional runner and no artifact round-trip.

The job binds JDK 8 deliberately, for the oldest supported `jni.h`. Gradle 9.6.1 needs JDK 17, so the
test steps add `setup-java` 17 and `setup-gradle` after the build and CRT check, exactly as the Linux
rows do.

`OpenVinoModelIT`'s parity bound stays at `atol=1e-2` and is **not** tightened.
`EtEngine.openVinoInferencePrecision()` reports `f32` or `bf16` depending on the host; both are
correct, and a tighter bound would assert which machine CI allocated rather than whether the
delegate works.

## 7. The risk that fails late

OpenVINO locates its CPU plugin at runtime, and the bundle ships no `plugins.xml` on either platform.
On Linux the flat directory resolves it through `RPATH=$ORIGIN`. On Windows the producer's loader
flags give the equivalent, and their own Windows correctness gate proves it — in *their* harness,
loading from *their* directory layout.

It is not yet proven from a JVM, loading out of our content-addressed cache directory, driven by our
`OPENVINO_LIB_PATH`. If that combination fails it fails late and unhelpfully: at first inference,
with `failed to import model for device 'CPU' (status=-1)`, while device enumeration still succeeds
— so a plugin-loading check would not catch it.

The plan therefore front-loads a manual Windows load-and-run smoke on winbox, against a hand-staged
bundle, **before** any Java refactoring. If the flat-directory assumption does not hold on Windows,
that is a bundle-layout problem for the producer and it invalidates §3 and §4 alike; discovering it
after the refactor would waste the whole of B.

## 8. Follow-ups

1. **The producer's `docs/openvino-jni-consumer.md` still says "Linux `x86_64` only."** Its selector
   section is current, but its library table, `setenv` recipe and checklist are all `.so`-shaped,
   while the Windows shape lives in `docs/handover-to-engine.md` C10. File upstream so the recipe doc
   and C10 stop disagreeing.
2. **General Windows JVM CI.** This spec adds `openvinoTest` on Windows; the wider gap is that no
   Windows JVM test runs in CI at all, so `gradlew.bat test` remains a manual winbox step. Worth
   deciding separately, since it is a Windows-CI question rather than an OpenVINO one.
