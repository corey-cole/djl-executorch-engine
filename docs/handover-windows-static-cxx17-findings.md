# Windows `/MT` adoption — measured findings

Spike run 2026-07-18 against runtime pin `v1.3.1-8`, on `winbox`: **VS 18 Community**,
MSVC 19.51.36248 (cl 14.51.36231), Ninja, Windows 10.0.26200.

Winbox is an iteration host, not the acceptance gate — CI runs VS 2022/17 Enterprise, whose
compiler defaults differ (see S4). **Re-confirmed by the Windows CI job before merge:** _pending._

Method: configured with the pin row and CRT overridden on the command line
(`-DET_PLATFORM=windows-x86_64-static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`); no source
changes. Every build directory was removed first — `CMAKE_MSVC_RUNTIME_LIBRARY` is a cache
variable, so a warm tree silently retains its old value and measures nothing.

## Stream A

| ID | Question | Result |
|----|----------|--------|
| S1 | Does the QA path link all-`/MT` with Catch2 via FetchContent? | **PASS** |
| S2 | Does the DLL still import `VCRUNTIME140`/`MSVCP140`? | **PASS** — no CRT imports |
| S3 | Catch2 suite + `./gradlew test` green? | **PASS** — both |
| S5 | What does `dumpbin -directives` emit for an import library? | **Confirmed**: nothing |

### S1 — Catch2 inherits the static CRT

The global `CMAKE_MSVC_RUNTIME_LIBRARY` cache variable **does** propagate into a FetchContent
subproject, which is what the design depends on. Directive counts:

| lib | LIBCMT | libcpmt | MSVCRT/MSVCPRT |
|---|---|---|---|
| `Catch2.lib` | 106 | 101 | **0** |
| `Catch2Main.lib` | 1 | 1 | **0** |

This validates choosing the global variable over the work order's A2 per-target
`set_property`, which cannot reach targets the project does not declare.

Note the markers are genuinely mixed-case (`LIBCMT` but `libcpmt`), so any scan must match
case-insensitively.

### S2 — the redistributable is gone

`dumpbin -dependents` on the `/MT` `executorch_djl.dll` lists exactly one dependency:

```
KERNEL32.dll
```

Contrast build against the `/MD` row, same source, to prove the check can fail:

```
MSVCP140.dll
VCRUNTIME140.dll
VCRUNTIME140_1.dll
VCRUNTIME140_THREADS.dll
```

A "no CRT imports" result is therefore meaningful rather than vacuous.

### S3 — it still works

- Catch2 suite: **22 assertions in 6 test cases, all passed.** This is also the Windows
  substitute for the `nm`-based XNNPACK registration guard, which cannot run on MSVC — the
  suite executes an XNNPACK-delegated `add.pte`, so a passing run proves the backend
  registered under `/MT`.
- JVM suite against the `/MT` DLL: **19 suites, 64 tests, 0 failures, 0 errors, 1 skipped.**
  The skip is `LstmModelIT`, expected: the `etnp::lstm` op ships only in the Linux `logging`
  tarball, and configure confirms `etnp extras: none in runtime (platform=windows-x86_64-static)`.

Both numbers required forcing real execution. Gradle reported `BUILD SUCCESSFUL` twice while
running nothing (`12 up-to-date`, then `:test FROM-CACHE`); the figures above come from a
`--rerun-tasks --no-build-cache` run with result XML timestamps checked against the wall clock.

### S5 — import libraries carry no CRT opinion

Directly measured, because the CRT scan's classification depends on it:

| file | kind | `dumpbin -directives` rc | `DEFAULTLIB` count |
|---|---|---|---|
| `executorch_djl.lib` | import library | 0 | **0** |
| `et_runtime.lib` | static library | 0 | 3 (`LIBCMT`, `libcpmt`, +1) |

The import library dumps only `.idata$*` / `.debug$S` sections. A two-way scan that requires
every `.lib` to carry a static marker therefore **fails on a correct build**. The gate's
three-way classification (tool failure → FAIL, no `DEFAULTLIB` at all → SKIP and count,
otherwise → must be static) is necessary, not defensive padding.

## Stream B

| ID | Question | Result |
|----|----------|--------|
| S4 | Which `/std:` flag reaches the ET-including TUs on MSVC? | **`-std:c++20`** — dash form |

The build emits `-std:c++20` (three occurrences in the verbose log), alongside `-MT`.

**The work order's B5 command is wrong and must not be used as written.** It greps for
`/std:c++`, but CMake's Ninja generator writes MSVC flags in dash form, so the pattern cannot
match. It returned nothing here on a build that plainly does set C++20 — a false negative that
reads exactly like the failure it was meant to detect. Correct form:

```bash
cmake --build <dir> --clean-first -- -v 2>&1 | grep -oE '[-/]std:c\+\+[0-9a-z]*' | sort -u
```

This is the same class of error as `dumpbin /nologo` vs `-nologo`, relocated from the tool
invocation into the verification pattern.

The flag is load-bearing, not decorative: MSVC 19.51 reports `_MSVC_LANG=201402` with no `/std`
flag, and `202002` with `-std:c++20`. So the compiler default is **C++14** and would trip
ExecuTorch's `#error` guard on its own — `set(CMAKE_CXX_STANDARD 20)` in `native/CMakeLists.txt`
is what satisfies it. (`__cplusplus` stays `199711` in both cases without `/Zc:__cplusplus`,
which is why ET's guard keys on `_MSVC_LANG` instead.)

## Not verified by execution

Loading the DLL on a Windows image that has never had a VC++ redistributable installed. No such
machine is available, and winbox has Visual Studio on it. The "no redist required" claim rests on
S2's import-table evidence.

## Note for the producer repo

Two items to report back, neither requiring an upstream `pytorch/executorch` issue from here:

1. The B5 command in the work order (`grep -o '/std:c++...'`) is a false negative under the Ninja
   generator, which emits dash-form MSVC flags.
2. Independently confirmed from the engine side that the installed ExecuTorch CMake package
   exports no `INTERFACE_COMPILE_FEATURES`, so `find_package` communicates no language standard —
   consistent with the producer repo's own finding and its drafted upstream proposal.
