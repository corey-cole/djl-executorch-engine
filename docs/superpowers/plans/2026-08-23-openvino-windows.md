# OpenVINO on windows-x86_64 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the OpenVINO runtime bundle for `windows-x86_64` as an opt-in capability jar, so a Windows consumer can run an OpenVINO-delegated `.pte` the same way a `linux-x86_64` consumer already can.

**Architecture:** The delegate already links on Windows and the pin already resolves a Windows bundle URL — the block is that `OpenVinoRuntime` reconstructs `.so`-and-ABI-shaped filenames that do not exist on Windows. Rather than branching on platform in Java, the staged bundle declares its own contents in `MANIFEST` and the extractor copies those names verbatim, which removes the ABI concept from the Java path entirely. The independent enumeration stays in the shell staging test, because a manifest generated from a truncated bundle would describe that truncation accurately.

**Tech Stack:** Java 17 (`OpenVinoRuntime`), Bash (`native/build.sh`, `native/tests/*.sh`), Catch2 v3.15.1 (native units), Gradle 9.6.1 (variant/capability publication), GitHub Actions (`windows-2022` runner, MSVC + Git-Bash).

**Spec:** `docs/superpowers/specs/2026-08-23-openvino-windows-design.md`

## Global Constraints

- **Branch:** `feature/openvino-windows`, already created. Never commit to `main`.
- **OpenVINO version is `2025.4.1` and does not move in this work.** Any step that looks like a version bump is wrong; `docs/openvino-version-bump.md` is a separate procedure.
- **`atol=1e-2` in the parity test is never tightened.** `EtEngine.openVinoInferencePrecision()` reports `f32` or `bf16` depending on the host; both are correct, and a tighter bound would assert which machine CI allocated.
- **The Windows bundle is six unversioned DLLs** — `openvino_c.dll`, `openvino.dll`, `openvino_intel_cpu_plugin.dll`, `openvino_ir_frontend.dll`, `tbb12.dll`, `tbbbind_2_5.dll` — with **four** `licenses/` files and **no `ov_abi` key in `BUILDINFO`**. Linux is seven ABI-versioned libraries, five licence files, and `ov_abi` present.
- **No symlink is ever created or required.** Jars do not carry symlinks; `OPENVINO_LIB_PATH` names the library file directly.
- **`native/cmake/EtRuntimePin.cmake` is generated. Never hand-edit it.**
- **No `sudo` in any step.**
- **Test convention:** `native/tests/*.sh` assert behaviour and policy bans, never the current wording of a message.
- **Comment convention:** comments state what *is*, not what *was*.
- **Out of scope:** `linux-aarch64` (links the delegate, no bundle published — the deliberate third state); the `example/` JMH OpenVINO comparison; any OpenVINO version change.

## Where each task runs

Only two tasks touch Windows. Everything else is Linux, and several steps below deliberately use
Linux-only paths and tools — that is not an oversight, it is the location.

| Task | Runs on | Why |
|---|---|---|
| 1 Steps 1-4 | Linux | Write the Catch2 case and prove it works where a bundle already exists |
| 1 Step 5 | **winbox** | The actual risk: does a flat directory resolve plugins on Windows |
| 2 | Linux | `build.sh` MANIFEST generation, verified against the Linux bundle |
| 3 | Linux | Java extractor rewrite |
| 4 | Linux | Pure-function test of an error message |
| 5 | Linux | `build.sh` support set and shell-test parameterization |
| 6 | Linux | Workflow YAML edit |
| 7 | Linux | Gradle variant proof against a synthetic tree |
| 8 | Linux | Documentation |
| 9 | **winbox** | Build, stage, and run the JVM suites on the real platform |
| 10 | Linux | Regression gate, issues, PR |

A Linux step that names a `.so`, uses `native/local_build_wrapper.sh`, or reads
`src/main/resources/native/linux-x86_64/` is correct **because it is a Linux step**. Windows steps
are marked `**On winbox:**` and use `.dll` paths and PowerShell.

## Working on winbox

Tasks 1 and 9 run on the Windows box. These are recorded constraints, not preferences — each one
has already cost a debugging session.

**Getting the branch there.** Push it and fetch on winbox:

```bash
git push -u origin feature/openvino-windows     # from the Linux box, after Task 1 Step 4
```

```powershell
$repo = "<the djl-executorch-engine checkout path on winbox>"
git -C $repo fetch origin
git -C $repo checkout feature/openvino-windows
git -C $repo reset --hard origin/feature/openvino-windows
```

Task 6 needs the branch pushed for CI regardless, so this costs nothing extra. If a work-in-progress
branch on `origin` is unwanted, the fallback is `git bundle create /tmp/ovwin.bundle feature/openvino-windows`
plus `scp`, then `git fetch /path/to/ovwin.bundle feature/openvino-windows`. **There is no rsync on
winbox** — `scp` with absolute forward-slash targets is the only copy mechanism.

**winbox needs no `executorch-runtime-dist` checkout.** Nothing in the build reads one: CMake
`FetchContent`s the ExecuTorch tarball by URL from the pin, and `build.sh` `curl`s the OpenVINO
bundle by URL from the pin. The clone at `~/workspace/executorch-runtime-dist` on the Linux box is
for reading producer documentation only. winbox does need **network access** to GitHub releases for
both downloads.

**Nothing may depend on the working directory.** A remote `pwsh` session starts in the user's
**home directory**, not the checkout. `Launch-VsDevShell.ps1` then changes the location again unless
given `-SkipAutomaticLocation`. And `Set-Location` is *not* a dependable fix for a native child
process: it changes PowerShell's provider location, which is a different thing from the process
working directory a launched `.exe` inherits. That is not a distinction worth discovering over SSH.

It matters because the two script families differ:

- `native/tests/*.sh` **root themselves** — each one does `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${REPO_ROOT}"`, so it only needs to be *found*.
- `native/build.sh` and `native/build_qa.sh` **do not.** They source `container_env.sh` relative to
  their own path and then use repo-relative paths (`native/build`, `src/main/resources/native/...`,
  `native/cmake/EtRuntimePin.cmake`) throughout. They must be *invoked from* the repo root. On Linux
  `local_build_wrapper.sh` guarantees that with `-w /workspace`; on Windows nothing does.

So make the `cd` explicit, in bash, where it is unambiguous. Write this helper locally once and
`scp` it over at the start of the session:

```bash
cat > /tmp/winbox-run.sh <<'EOF'
#!/usr/bin/env bash
# Usage: winbox-run.sh <repo-path-in-windows-form> <script> [args...]
# Runs a repo script from the repo root, so nothing depends on what working directory PowerShell
# handed the bash child. native/build.sh and native/build_qa.sh need this; native/tests/*.sh cd
# themselves but are harmless to run through it, so everything goes through one path.
set -euo pipefail
cd "$(cygpath -u "$1")"
shift
exec "$@"
EOF

# scp on this host REQUIRES an absolute, forward-slash target -- a bare relative path fails on its
# sftp server. Bind the checkout path in both forms once, locally:
#   REPO_FS=C:/Users/<user>/workspace/djl-executorch-engine
scp /tmp/winbox-run.sh "winbox:${REPO_FS}/winbox-run.sh"
```

Landing the helpers inside the checkout keeps every path derived from one value. They are untracked,
so `git reset --hard` leaves them; delete them at the end of Task 9.

Bind the same path in PowerShell (backslash form) and the two helper handles once per session:

```powershell
$repo = "<the djl-executorch-engine checkout path on winbox>"
$bash = "${env:ProgramFiles}\Git\bin\bash.exe"
$run  = "$repo\winbox-run.sh"
```

Each call is then one command with plain arguments — no `&&` chain, no inherited-cwd assumption:

```powershell
& $bash $run $repo ./native/build_qa.sh
```

The checkout path is machine-specific and deliberately not recorded in this repo — the winbox
hostname, user, and key path live in `windows-jni-handoff.md`, outside version control.

Gradle gets the same treatment through its own flag rather than the helper: invoke the wrapper by
absolute path and pass `-p $repo`, so the project directory is stated rather than inferred.

Both helpers live in the checkout rather than `/tmp`: Windows OpenSSH and Git-Bash do not agree on
what `/tmp` means, and this host's sftp server rejects relative `scp` targets anyway.

**JDK.** `gradlew.bat` in Task 9 needs a JDK 17; winbox has Zulu 17.0.19, recorded as confirmed
present in `docs/superpowers/plans/2026-08-09-production-observability.md`. (An older plan,
`2026-07-18-windows-static-crt.md`, says winbox may have no JVM toolchain — that is superseded, and
the contradiction is why the follow-up issue in Task 10 exists.) This is separate from the JDK the
shim compiles against: `build.sh` needs only `JAVA_HOME` pointing at any JDK for `jni.h`, and never
links `libjvm`.

**Driving it.**

- `ssh winbox` is key-based and lands directly in **PowerShell 7 Core** (`pwsh`), so the blocks
  below are typed as-is — no `-EncodedCommand` wrapping. Write PowerShell 7 syntax (`$env:VAR`,
  `Test-Path`, `Get-ChildItem -Name`); `cmd` syntax fails outright. Windows PowerShell 5.1 is also
  installed and is **not** what you get: anything invoked as `powershell.exe -Command ...` runs 5.1
  instead, where `&&` and `?:` do not exist and `>` writes UTF-16LE rather than UTF-8.
- **A `bash -c '...'` one-liner does not survive PowerShell→native-exe quoting.** Keep every
  Git-Bash invocation to one simple command with plain arguments, or write a `.sh` file and invoke
  that. Never chain with `&&` inside `bash -c`.
- Invoke Git-Bash by explicit path (`${env:ProgramFiles}\Git\bin\bash.exe`); a bare `bash` can
  select WSL's `System32\bash.exe`. Use `-c`, never `-lc` — a login shell resets PATH and drops the
  VS environment.
- Redirect stdin on the **local `ssh` invocation** — `ssh winbox ... < /dev/null` — so nothing can
  block waiting for input. It does not go inside the remote command: PowerShell has no `<`
  redirection operator and answers `The '<' operator is reserved for future use.`
- Prefer several short calls over one long one: from outside, a blocked run and a slow one look
  identical, and a first `gradlew` run silently downloading a distribution looks exactly like a hang.
- Run Gradle as `& "$repo\gradlew.bat" -p $repo --no-daemon --console=plain <task>`: absolute wrapper
  path and an explicit project directory, for the same reason as the helper. `--no-daemon` matters
  independently — a lingering daemon looks like a stuck process to whoever is watching the box.
- **winbox is for iteration, not acceptance.** It runs VS 18 Community against the runner's VS 17
  Enterprise. The `windows-2022` runner is the sole acceptance gate — which is why Task 6 exists and
  why Task 9 is verification rather than sign-off.

---

### Task 1: Prove the flat-directory bundle loads on Windows

This is the spec's front-loaded risk (§7) and it runs **before any refactoring**. OpenVINO locates its CPU plugin at runtime and the bundle ships no `plugins.xml`; on Linux `RPATH=$ORIGIN` resolves it, and on Windows the producer's `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` is supposed to. That is proven in the producer's harness, not from a flat directory we extracted. If it does not hold, §3 and §4 of the spec are both void and the fix belongs in the producer's bundle layout.

The existing Catch2 OpenVINO cases only assert **refusals** — unset path, directory path. There is no case proving a *successful* load, on either platform. This task adds one, which is a real coverage gap independent of Windows.

**Location:** Steps 1-4 on **Linux**; Step 5 on **winbox**. The Linux half proves the case is real
against the bundle that already ships there; the Windows half is the risk being retired.

**Files:**
- Modify: `native/test/et_runtime_test.cpp` (append at end of file)
- Modify: `native/local_build_wrapper.sh` (add one `-e` passthrough)

**Interfaces:**
- Produces: a Catch2 case gated on the environment variable `ET_OPENVINO_SMOKE_LIB`. A distinct variable name is required: the existing guard cases call `unsetEnvVar("OPENVINO_LIB_PATH")`, so anything read from `OPENVINO_LIB_PATH` at case-run time would already be gone.

- [ ] **Step 1: Write the failing test**

Append to `native/test/et_runtime_test.cpp`. It must go at the **end** of the file: Catch2 runs in registration order, and the intra-op pool cases at the top require that no `EtRuntime` has been constructed yet.

```cpp
// The only case that proves a SUCCESSFUL OpenVINO load. Everything else in this file asserts the
// guard's refusals, which pass just as well when the runtime cannot actually resolve its plugins.
//
// Gated on ET_OPENVINO_SMOKE_LIB rather than OPENVINO_LIB_PATH because the guard cases above
// unsetEnvVar("OPENVINO_LIB_PATH"), so a value the operator exported would be gone by the time this
// runs. Point it at the C library file inside an extracted bundle:
//   ET_OPENVINO_SMOKE_LIB=/path/to/bundle/lib/libopenvino_c.so.2541   (linux)
//   ET_OPENVINO_SMOKE_LIB=C:\path\to\bundle\lib\openvino_c.dll        (windows)
//
// What this really tests is the FLAT DIRECTORY assumption: the bundle carries no plugins.xml, so
// the CPU plugin and the IR frontend must be found as siblings of the C library. A failure here
// surfaces as "failed to import model for device 'CPU'" while device enumeration still succeeds,
// which is why a plugin-loading check would not catch it.
TEST_CASE("openvino: a bundle in one flat directory loads and executes") {
#ifndef ET_OPENVINO_LINKED
  SKIP("this build links no OpenVINO delegate");
#else
  const char* smoke = std::getenv("ET_OPENVINO_SMOKE_LIB");
  if (smoke == nullptr || smoke[0] == '\0') {
    SKIP("set ET_OPENVINO_SMOKE_LIB to a bundle's OpenVINO C library file");
  }
  setEnvVar("OPENVINO_LIB_PATH", smoke);
  EtRuntime rt(OPENVINO_TINY_PTE_PATH);
  const auto meta = rt.methodMeta();
  REQUIRE(meta.numInputs > 0);
  unsetEnvVar("OPENVINO_LIB_PATH");
#endif
}
```

`EtRuntime::methodMeta()` returns a `MethodMeta` with a `numInputs` field (`native/core/et_runtime.h:34,89`); other cases in this file read it the same way. The assertion's only job is to prove construction completed — construction is where `load_forward()` runs delegate init, and therefore where plugin resolution happens.

- [ ] **Step 2 (Linux): Let the wrapper pass the variable through**

`native/build_qa.sh` runs the whole Catch2 suite itself (`./native/asan/et_runtime_test --order decl`),
so the smoke case needs no hand-invocation — it just needs its variable to exist inside the
container. `local_build_wrapper.sh` forwards a **fixed allowlist** of `-e` flags, so add one beside
the others:

```bash
    -e ET_OPENVINO_SMOKE_LIB \
```

Running the binary on the host instead is not an alternative: it is built in the container against
that image's ASan runtime.

- [ ] **Step 3 (Linux): Run it — skip, then fail, then pass**

First with the variable unset:

```bash
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: the new case reports SKIP and the rest of the suite is green.

Then prove it can fail, by pointing it at a bundle with a library removed. The path must be inside
the repo — the wrapper bind-mounts `$REPO_ROOT` and nothing else, so a `/tmp` path would not exist
in the container:

```bash
rm -rf native/build/ov-broken && cp -r src/main/resources/native/linux-x86_64/openvino/lib native/build/ov-broken
rm native/build/ov-broken/libopenvino_ir_frontend.so.*
ET_OPENVINO_SMOKE_LIB="/workspace/native/build/ov-broken/$(ls native/build/ov-broken | grep '^libopenvino_c\.so\.')" \
  ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: FAIL with an import error naming device `CPU`. Note the `/workspace/...` prefix: the
variable is read inside the container, where the repo is mounted at `/workspace`.

Then against the intact bundle:

```bash
ET_OPENVINO_SMOKE_LIB="/workspace/src/main/resources/native/linux-x86_64/openvino/lib/$(ls src/main/resources/native/linux-x86_64/openvino/lib | grep '^libopenvino_c\.so\.')" \
  ./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: PASS. Then `rm -rf native/build/ov-broken`.

- [ ] **Step 4 (Linux): Commit**

```bash
git add native/test/et_runtime_test.cpp native/local_build_wrapper.sh
git commit -m "test(native): prove an OpenVINO bundle loads from one flat directory

Every other OpenVINO case here asserts a refusal, which passes just as well
when the runtime cannot resolve its plugins. This is the case that fails if
the flat-directory assumption breaks -- which surfaces as an import error at
first inference while device enumeration still succeeds.

Gated on ET_OPENVINO_SMOKE_LIB, not OPENVINO_LIB_PATH: the guard cases above
unset the latter, so an exported value would be gone by the time this runs."
```

- [ ] **Step 5 (winbox): Run it on Windows — this is the go/no-go**

Push the branch and check it out on winbox (see "Working on winbox" above). Then write this
throwaway script **locally** and `scp` it over, rather than pasting a command chain — a chained
`bash -c` does not survive the PowerShell→native-exe quoting:

```bash
cat > /tmp/ov-smoke-stage.sh <<'EOF'
#!/usr/bin/env bash
# Throwaway: stages the windows-x86_64 OpenVINO bundle outside the build, so the flat-directory
# assumption can be tested before any engine code depends on it. Prints the C library path.
set -euo pipefail
PIN=native/cmake/EtRuntimePin.cmake
url="$(grep -oPz 'set\(ET_RUNTIME_OPENVINO_URL_windows-x86_64\s+"\K[^"]+' "$PIN" | tr -d '\0')"
sha="$(grep -oPz 'set\(ET_RUNTIME_OPENVINO_SHA256_windows-x86_64\s+"\K[^"]+' "$PIN" | tr -d '\0')"
[ -n "$url" ] || { echo "no windows-x86_64 OpenVINO row in the pin"; exit 1; }
rm -rf /tmp/ovwin && mkdir -p /tmp/ovwin/b
curl -fsSL -o /tmp/ovwin/bundle.tar.gz "$url"
echo "$sha  /tmp/ovwin/bundle.tar.gz" | sha256sum -c -
tar xzf /tmp/ovwin/bundle.tar.gz --strip-components=1 -C /tmp/ovwin/b
echo "--- staged libraries (expect exactly six DLLs) ---"
ls -1 /tmp/ovwin/b/lib
# A Windows-style absolute path: the producer's is_absolute_path checks for a drive letter or a
# leading separator, and a Git-Bash /tmp/... path is neither, so LoadLibraryExW would take the
# bare-filename branch and search somewhere else entirely.
echo "SMOKE_LIB=$(cygpath -w /tmp/ovwin/b/lib/openvino_c.dll)"
EOF
scp /tmp/ov-smoke-stage.sh "winbox:${REPO_FS}/ov-smoke-stage.sh"   # absolute, forward slashes
```

Stage the bundle, as its own short call with stdin redirected. It reads
`native/cmake/EtRuntimePin.cmake` by relative path, so it goes through `winbox-run.sh` like
everything else — `$repo`, `$bash` and `$run` are the bindings from "Working on winbox":

```powershell
& $bash $run $repo "$repo\ov-smoke-stage.sh"
```

Expected: exactly six DLLs listed, and a `SMOKE_LIB=C:\...\openvino_c.dll` line.

Then set that value and run the QA build. `build_qa.sh` builds and runs the whole Catch2 suite
itself (`./native/asan/et_runtime_test.exe --order decl`), so the smoke case runs as part of it —
there is no binary to invoke by hand, and no sanitizers on this platform despite the directory name:

```powershell
$env:ET_OPENVINO_SMOKE_LIB = "<the SMOKE_LIB value printed above>"
& $bash $run $repo ./native/build_qa.sh
```

Expected: the suite green, including `openvino: a bundle in one flat directory loads and executes`
as a PASS rather than a skip. If it reports a skip, `ET_OPENVINO_SMOKE_LIB` did not reach the
Git-Bash child — check it with `& $bash -c 'echo $ET_OPENVINO_SMOKE_LIB'`.

**If this fails, stop.** Capture the exact error and `GetLastError` value. An import failure means the flat-directory layout does not resolve plugins on Windows, which is a producer bundle-layout issue and invalidates the rest of this plan — report it rather than working around it in Java.

---

### Task 2: The bundle declares its own contents

**Location:** Linux. The code being edited also *runs* on Windows under Git-Bash, which is why the listing below avoids `find -printf`.

**Files:**
- Modify: `native/build.sh:216-222` (the symlink removal and `MANIFEST` write inside the `stage)` arm)
- Test: `native/tests/openvino_bundle_staging.sh`

**Interfaces:**
- Produces: two new `MANIFEST` keys read by Task 3.
  - `libs` — space-separated filenames of everything in the staged `lib/`, sorted. No OpenVINO library filename contains a space on either platform.
  - `c_library` — the single filename of the OpenVINO C API library (`libopenvino_c.so.<abi>` on Linux, `openvino_c.dll` on Windows).

- [ ] **Step 1: Write the failing test**

Append to `native/tests/openvino_bundle_staging.sh`, before its final `echo "PASS: ..."` line:

```bash
# The bundle declares its own contents so the Java extractor copies names rather than reconstructing
# them -- which is what lets one code path serve an ABI-versioned Linux bundle and an unversioned
# Windows one. Asserted against the actual lib/ directory, not just for presence: a MANIFEST that
# disagrees with the tree would send the extractor after a file that is not there, failing at
# dlopen rather than here.
man_libs="$(grep -oP '^libs=\K.*' "${DIR}/MANIFEST" || true)"
[ -n "${man_libs}" ] || fail "MANIFEST carries no libs"

actual_libs="$(ls -1 "${DIR}/lib" | sort | tr '\n' ' ')"
[ "${man_libs} " = "${actual_libs}" ] \
  || fail "MANIFEST libs disagree with lib/: manifest='${man_libs}' actual='${actual_libs%% }'"

man_clib="$(grep -oP '^c_library=\K.*' "${DIR}/MANIFEST" || true)"
[ -n "${man_clib}" ] || fail "MANIFEST carries no c_library"
[ -f "${DIR}/lib/${man_clib}" ] || fail "c_library names a file that is not in lib/: ${man_clib}"
# It must be the C API library specifically -- pointing OPENVINO_LIB_PATH at any other library in
# the bundle loads something that resolves no ov_* symbols.
case "${man_clib}" in
  libopenvino_c.so.*|openvino_c.dll) ;;
  *) fail "c_library is not the OpenVINO C API library: ${man_clib}" ;;
esac
```

- [ ] **Step 2: Run it to verify it fails**

```bash
./native/tests/openvino_bundle_staging.sh
```

Expected: FAIL with `MANIFEST carries no libs`. If it reports `SKIP: bundle not staged`, run `./native/local_build_wrapper.sh` first and copy the staged tree into place the way CI does:

```bash
mkdir -p build/native-staging/linux-x86_64
cp -r src/main/resources/native/linux-x86_64/. build/native-staging/linux-x86_64/
```

- [ ] **Step 3: Write the MANIFEST keys in `native/build.sh`**

In the `stage)` arm, replace the symlink-removal line and the `MANIFEST` write block with:

```bash
    # Linux only: the bundle ships an unversioned compatibility symlink beside the versioned file.
    # Jars do not preserve symlinks and nothing needs it -- OPENVINO_LIB_PATH names the versioned
    # file directly and $ORIGIN resolves the rest. Guarded rather than left to `rm -f` missing on
    # Windows, so the intent survives someone reading only this line.
    if [ -L "${OV_OUT}/lib/libopenvino_c.so" ]; then
      rm -f "${OV_OUT}/lib/libopenvino_c.so"
    fi

    # The bundle declares its own contents. The Java extractor copies these names verbatim instead
    # of reconstructing them from an ABI suffix, which is what lets one code path serve an
    # ABI-versioned Linux bundle and an unversioned Windows one -- the Windows BUILDINFO carries no
    # ov_abi key at all. Generated from what actually landed in lib/, so it cannot disagree with the
    # tree. Computed AFTER the symlink removal above, or the symlink would be listed.
    # ls -1, not find -printf: this runs under Git-Bash on Windows too, and lib/ is flat by
    # contract -- the staging test asserts it.
    OV_LIBS="$(ls -1 "${OV_OUT}/lib" | sort | tr '\n' ' ')"
    OV_LIBS="${OV_LIBS% }"
    [ -n "${OV_LIBS}" ] || { echo "staged OpenVINO bundle has no libraries"; exit 1; }

    OV_CLIB="$(printf '%s\n' ${OV_LIBS} | grep -E '^(lib)?openvino_c\.(so|dll)' | head -1)"
    [ -n "${OV_CLIB}" ] || { echo "no OpenVINO C API library in the staged bundle"; exit 1; }

    {
      echo "openvino_version=${OV_VER}"
      echo "tarball_sha256=${OV_SHA}"
      echo "tarball_url=${OV_URL}"
      echo "libs=${OV_LIBS}"
      echo "c_library=${OV_CLIB}"
    } > "${OV_OUT}/MANIFEST"
```

`printf '%s\n' ${OV_LIBS}` is deliberately unquoted so the space-separated list splits into lines for `grep`.

- [ ] **Step 4: Restage and run the test to verify it passes**

```bash
./native/local_build_wrapper.sh
mkdir -p build/native-staging/linux-x86_64
cp -r src/main/resources/native/linux-x86_64/. build/native-staging/linux-x86_64/
./native/tests/openvino_bundle_staging.sh
cat src/main/resources/native/linux-x86_64/openvino/MANIFEST
```

Expected: PASS, and the MANIFEST shows seven space-separated Linux filenames plus `c_library=libopenvino_c.so.2541`.

- [ ] **Step 5: Commit**

```bash
git add native/build.sh native/tests/openvino_bundle_staging.sh
git commit -m "feat(build): have the OpenVINO bundle declare its own contents

libs and c_library are generated from the staged lib/ directory, so the
extractor can copy names instead of reconstructing them from an ABI suffix.
That is what lets one code path serve an ABI-versioned Linux bundle and an
unversioned Windows one, whose BUILDINFO carries no ov_abi key at all.

The staging test keeps its own independent enumeration: a MANIFEST generated
from a truncated bundle would describe that truncation accurately."
```

---

### Task 3: `OpenVinoRuntime` reads the manifest instead of reconstructing names

**Location:** Linux.

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java` — delete `LIBS` (lines 44-52), rewrite `publish()`'s copy loop, rewrite `resolvedLibPath()`, delete `buildInfo()`
- Test: `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java:52`

**Interfaces:**
- Consumes: `MANIFEST` keys `libs` and `c_library` from Task 2.
- Produces: `resolvedLibPath()` returns the absolute path of the file named by `c_library` inside the extracted directory. `publish()` copies exactly the files named by `libs`, plus `BUILDINFO`.

- [ ] **Step 1: Update the failing assertion**

`OpenVinoRuntimeTest:52` currently asserts `lib.contains(".so.")`, which is false on Windows. Replace it with an assertion that the resolved path is the library the bundle declared — platform-free, and a stronger statement than the string check it replaces:

```java
        // The library the BUNDLE declared, not one this test reconstructs: that is the whole point
        // of c_library, and it is what makes this assertion identical on Windows.
        java.util.Properties man = new java.util.Properties();
        try (var is = OpenVinoRuntime.class.getResourceAsStream(
                "/native/" + LibUtils.platform() + "/openvino/MANIFEST")) {
            man.load(new java.io.InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8));
        }
        assertEquals(dir.resolve(man.getProperty("c_library")).toAbsolutePath().toString(), lib);
        assertFalse(Files.isSymbolicLink(Paths.get(lib)), "must never resolve through a symlink");
```

Add `import static org.junit.jupiter.api.Assertions.assertFalse;` if it is not already present.

- [ ] **Step 2: Run it to verify it fails**

```bash
./gradlew openvinoTest --tests 'org.measly.executorch.engine.OpenVinoRuntimeTest'
```

Expected: FAIL — `c_library` resolves to `null` inside `dir.resolve(...)` and throws `NullPointerException`, because the staged MANIFEST in the *test* classpath predates Task 2 unless you restaged. If it passes immediately you restaged in Task 2 and the MANIFEST already has the key; in that case temporarily delete the `c_library=` line from `src/main/resources/native/linux-x86_64/openvino/MANIFEST`, confirm the failure, and restore it.

- [ ] **Step 3: Rewrite the extractor**

Delete the `LIBS` constant entirely. Add two accessors beside the existing `manifest()` helper:

```java
    /** @return the library filenames the staged bundle declares, in the order it listed them */
    private static List<String> bundleLibs() {
        String libs = manifest().getProperty("libs");
        if (libs == null || libs.isBlank()) {
            throw new IllegalStateException(
                    "OpenVINO bundle MANIFEST carries no libs; restage with native/build.sh");
        }
        return List.of(libs.trim().split("\\s+"));
    }

    /** @return the filename of the OpenVINO C API library the delegate must dlopen */
    private static String bundleCLibrary() {
        String name = manifest().getProperty("c_library");
        if (name == null || name.isBlank()) {
            throw new IllegalStateException(
                    "OpenVINO bundle MANIFEST carries no c_library; restage with native/build.sh");
        }
        return name;
    }
```

Replace `publish()`'s copy loop — the whole `for (String lib : LIBS) { ... }` block including its ABI-suffix comment and resource-exists fallback — with:

```java
            // The bundle names its own files, so nothing here reconstructs a versioned filename.
            // That is what removes the ABI concept from this path: the Windows bundle ships six
            // unversioned DLLs and carries no ov_abi key at all.
            for (String lib : bundleLibs()) {
                copy(resourceBase() + "lib/" + lib, staging.resolve(lib));
            }
```

Replace `resolvedLibPath()`'s body after the two early returns:

```java
        libPath = extracted.resolve(bundleCLibrary()).toAbsolutePath().toString();
        return libPath;
```

Delete the now-unused `buildInfo()` method. Keep the `BUILDINFO` constant and the `copy(resourceBase() + BUILDINFO, staging.resolve(BUILDINFO))` line in `publish()` — the file is still copied into the cache directory as provenance.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./gradlew openvinoTest
```

Expected: green, including `OpenVinoModelIT` and `OpenVinoConcurrentExtractionTest`. The concurrent test exercises `publish()` directly, so it covers the new copy loop.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java \
        src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java
git commit -m "refactor(openvino): extract the libraries the bundle declares

LIBS, the ov_abi suffix construction, and the resource-exists fallback for
the one library that ships unversioned all go away together: the staged name
is simply the name. ov_abi is no longer read by Java at all, which is what
makes a Windows bundle work rather than a special case tolerating its
absence."
```

---

### Task 4: Correct the no-delegate error branch and cover it

**Location:** Linux.

**Files:**
- Modify: `src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java:94-104` (the `!EtNative.backendRegistered(BACKEND)` branch)
- Modify: `src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java` — `validateOverride`'s directory message
- Test: `src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java`

The branch's comment says "No delegate in this build at all -- Windows today". That is now false, and it is exactly how a future reader concludes Windows has no delegate. The branch itself stays: it is still reachable through the `ET_INSTALL` escape hatch, which links a caller-supplied runtime tree that may have been built without OpenVINO.

**The condition is trivial and the message is the asset**, so the message is what gets tested. Extracting it into a pure function makes it testable on every platform, which an assumption-gated test guarding `!backendRegistered(...)` would not be — that assumption is false on all three shipped platforms, so such a test would skip everywhere and prove nothing.

- [ ] **Step 1 (Linux): Write the failing test**

Add to `OpenVinoRuntimeTest`. It calls the message builder directly, so it needs no native state, no
bundle, and no assumption — it runs and asserts on every platform:

```java
    @Test
    void theNoDelegateErrorDirectsTheUserToReExport() {
        // The condition guarding this message (!backendRegistered) is false on every SHIPPED
        // platform: all three runtime tarballs carry the delegate. It stays reachable through the
        // ET_INSTALL escape hatch, which links a caller-supplied runtime tree that may have been
        // built without OpenVINO -- so the message must stay correct, and a test gated on the
        // condition would skip everywhere and prove nothing. The message is the asset; test it.
        String msg = OpenVinoRuntime.noDelegateMessage().getMessage();
        assertTrue(msg.contains(OpenVinoRuntime.BACKEND), "must name the backend: " + msg);
        assertTrue(msg.contains(LibUtils.platform()), "must name the platform: " + msg);
        // The remedy is the whole point of keeping this distinct from the no-runtime error: one
        // says re-export the model, the other says add a runtime artifact. Asserting the remedy is
        // what stops the two from converging.
        assertTrue(msg.contains("Re-export"), "must direct the user to re-export: " + msg);
        assertFalse(
                msg.contains("-openvino artifact"),
                "must not offer the runtime artifact; no runtime can help here: " + msg);
    }
```

- [ ] **Step 2 (Linux): Run it to verify it fails**

```bash
./gradlew openvinoTest --tests 'org.measly.executorch.engine.OpenVinoRuntimeTest'
```

Expected: compilation failure — `noDelegateMessage()` does not exist.

- [ ] **Step 3 (Linux): Extract the message and correct its comment**

Replace the branch body with a call to a new package-private builder, and move the explanation onto
the builder where the message lives:

```java
        if (!EtNative.backendRegistered(BACKEND)) {
            throw noDelegateMessage();
        }
```

Add the builder beside `validateOverride`:

```java
    /**
     * The error for a build that links no OpenVINO delegate.
     *
     * <p>Every shipped runtime tarball carries the delegate, so in practice this means a runtime
     * tree supplied through the {@code ET_INSTALL} escape hatch that was built without it. No
     * runtime artifact can help — the model cannot execute here at all, so the only remedy is to
     * re-export. Deliberately distinct from the no-runtime error, which looks similar to a user and
     * has the opposite remedy.
     *
     * <p>Package-private and separate from its guard so it can be asserted directly: the guard is
     * false on every shipped platform, so a test gated on it would skip everywhere.
     *
     * @return the exception to throw
     */
    static EngineException noDelegateMessage() {
        return new EngineException(
                "This .pte uses the "
                        + BACKEND
                        + " delegate, which this build does not provide ("
                        + LibUtils.platform()
                        + "). The delegate ships only where the ExecuTorch runtime was built"
                        + " with it. Re-export without the OpenVINO partitioner to run here.");
    }
```

In `validateOverride`, the directory-mistake message hardcodes a `.so` example. Make it name the bundle's own library when one is present:

```java
        if (Files.isDirectory(candidate)) {
            // Upstream's documented top mistake, and an easy one to make: the error the delegate
            // would otherwise produce mentions LD_LIBRARY_PATH, which reads like it wants a
            // directory. It does not.
            String example = bundleAvailable() ? bundleCLibrary() : "the OpenVINO C library";
            throw new EngineException(
                    "OPENVINO_LIB_PATH points at a directory: '" + value + "'. It must be the full "
                            + "path to the library FILE itself, e.g. <dir>/" + example + ".");
        }
```

`bundleCLibrary()` is private and in the same class, so no visibility change is needed.

- [ ] **Step 4 (Linux): Run the tests to verify they pass**

```bash
./gradlew openvinoTest
```

Expected: green, with the new test now running rather than skipping. The existing `validateOverride`
directory test asserts on the message; if it greps for the old `.so` example text, update it to
assert the behaviour — that the message names a FILE and includes the offending value — rather than
the example string.

- [ ] **Step 5 (Linux): Commit**

```bash
git add src/main/java/org/measly/executorch/engine/OpenVinoRuntime.java \
        src/test/java/org/measly/executorch/engine/OpenVinoRuntimeTest.java
git commit -m "fix(openvino): state the real condition for the no-delegate error

The comment said 'Windows today', which the 1.4.1 runtime made false -- every
shipped tarball now carries the delegate. The branch stays because ET_INSTALL
can link a runtime tree built without it.

Its message moves into a package-private builder so it can be asserted
directly. Gating a test on the guard instead would skip on every shipped
platform, which is how a user-facing string rots.

The override message no longer offers a .so example on a platform that has no
.so files."
```

---

### Task 5: Stage the Windows bundle

**Location:** Linux. The Windows branch added here is *exercised* on winbox in Task 9 and in CI in Task 6; on this host it correctly reports `SKIP: bundle not staged`.

**Files:**
- Modify: `native/build.sh:33` (`ET_OPENVINO_SUPPORTED_PLATFORMS` default) and its comment at lines 28-32
- Modify: `native/tests/openvino_bundle_staging.sh` (platform parameter)
- Modify: `native/tests/openvino_version_coupling.sh:18` (platform parameter for the staged branch)
- Test: `native/tests/openvino_pin_selector.sh`

**Interfaces:**
- Consumes: `et_openvino_resolve` and the `OV_DECISION` values from sub-project A.
- Produces: `openvino_bundle_staging.sh [platform]` and `openvino_version_coupling.sh [platform]`, both defaulting to `linux-x86_64`.

- [ ] **Step 1: Update the failing selector test**

In `native/tests/openvino_pin_selector.sh`, the case asserting today's Windows state inverts. Replace:

```bash
# Today's Windows state: the pin PUBLISHES a bundle, the engine does not support it. Staging it would
# put ~21 MB in a jar nothing can load, because OpenVinoRuntime's library list is .so-shaped.
out="$(decide windows-x86_64)"
grep -q 'decision=unsupported' <<<"${out}" || fail "windows-x86_64 must be unsupported today: ${out}"
```

with:

```bash
# Windows is supported: the runtime tarballs ship openvino_backend.lib, the pin publishes a
# win_amd64 bundle, and OpenVinoRuntime extracts whatever the bundle's MANIFEST declares.
out="$(decide windows-x86_64)"
grep -q 'decision=stage' <<<"${out}" || fail "windows-x86_64 must stage: ${out}"
# The literal-URL guard. The pin expresses "both Windows CRT rows share one bundle" as an ALIAS row
# whose VALUE is a CMake variable reference. A row-keyed lookup would hand curl the unexpanded text
# ${ET_RUNTIME_OPENVINO_URL_windows-x86_64}; keying on the platform never reads the alias.
grep -q 'url=https://' <<<"${out}" || fail "windows url must be literal, not a cmake reference: ${out}"
```

Delete the now-redundant case immediately below it that flipped the supported set to reach the same conclusion — with Windows in the default set it asserts nothing new. Keep both `linux-aarch64` cases: they still prove the support and publication questions are independent.

- [ ] **Step 2: Run it to verify it fails**

```bash
./native/tests/openvino_pin_selector.sh
```

Expected: FAIL with `windows-x86_64 must stage`.

- [ ] **Step 3: Add Windows to the supported set**

In `native/build.sh`, replace the comment block and default at lines 28-33:

```bash
# The engine's OpenVINO support set: what the JAVA layer can load, which is not the same question as
# what the pin publishes. A platform belongs here once OpenVinoRuntime can extract its bundle and a
# test proves it -- adding it is the LAST step of supporting a platform, never the first.
# linux-aarch64 is absent because upstream publishes no bundle for it, not because of anything here.
ET_OPENVINO_SUPPORTED_PLATFORMS="${ET_OPENVINO_SUPPORTED_PLATFORMS:-linux-x86_64 windows-x86_64}"
```

- [ ] **Step 4: Run the selector test to verify it passes**

```bash
./native/tests/openvino_pin_selector.sh
```

Expected: `PASS: openvino pin selector`.

- [ ] **Step 5: Parameterize the two staged-tree shell tests**

In `native/tests/openvino_bundle_staging.sh`, replace the hardcoded `DIR` with a platform parameter and make the member expectations per-platform. Replace the `DIR=` line with:

```bash
# Platform-parameterized: CI runs this once per row that stages a bundle. Default keeps every
# existing caller working unchanged.
PLATFORM="${1:-linux-x86_64}"
DIR="build/native-staging/${PLATFORM}/openvino"
```

Replace the ABI derivation and the member/count block with:

```bash
# The expected member SET is the part worth reviewing on a version bump -- an OpenVINO release can
# add or drop a transitive dependency, and a missing one fails at model load with an error naming
# none of this. Kept independent of MANIFEST on purpose: a manifest generated from a truncated
# bundle would describe that truncation accurately. See docs/openvino-version-bump.md.
case "${PLATFORM}" in
  linux-x86_64)
    # The ABI suffix is DERIVED from BUILDINFO, never hardcoded: it tracks the OpenVINO version
    # (2025.4.1 -> 2541), so a literal would be a thing to edit on every bump.
    abi="$(grep -oP '^ov_abi=\K.*' "${DIR}/BUILDINFO")"
    [ -n "${abi}" ] || fail "BUILDINFO carries no ov_abi"
    expected="libopenvino_c.so.${abi} libopenvino.so.${abi} libopenvino_intel_cpu_plugin.so"
    expected="${expected} libopenvino_ir_frontend.so.${abi} libtbb.so.12 libtbbbind_2_5.so.3"
    expected="${expected} libhwloc.so.15"
    ;;
  windows-x86_64)
    # No ov_abi key at all here, and its ABSENCE is asserted rather than tolerated: the DLLs are
    # unversioned, so a bundle that grew one would mean the upstream layout changed under us.
    # `grep && fail` is safe under set -e -- a failing non-final member of an AND-list does not
    # exit, which is the same idiom docs_present.sh uses for its policy bans.
    grep -q '^ov_abi=' "${DIR}/BUILDINFO" && fail "windows BUILDINFO must carry no ov_abi"
    # Six, not seven: hwloc is folded into tbbbind_2_5.dll on Windows.
    expected="openvino_c.dll openvino.dll openvino_intel_cpu_plugin.dll openvino_ir_frontend.dll"
    expected="${expected} tbb12.dll tbbbind_2_5.dll"
    ;;
  *) fail "no expected library set for platform '${PLATFORM}'" ;;
esac

for f in ${expected}; do
  [ -f "${DIR}/lib/${f}" ] || fail "missing library: ${f} (see docs/openvino-version-bump.md)"
done

# Nothing may be shipped that no one enumerated: an unlisted library means the bundle grew and the
# expectations above have not caught up.
want="$(printf '%s\n' ${expected} | wc -l)"
count="$(ls -1 "${DIR}/lib" | wc -l)"
[ "${count}" -eq "${want}" ] \
  || fail "expected ${want} libraries, found ${count} -- the bundle changed; see docs/openvino-version-bump.md"
```

In `native/tests/openvino_version_coupling.sh`, replace the hardcoded staged path:

```bash
staged="build/native-staging/${1:-linux-x86_64}/openvino/MANIFEST"
```

- [ ] **Step 6: Verify both still pass on Linux**

```bash
./native/tests/openvino_bundle_staging.sh
./native/tests/openvino_bundle_staging.sh linux-x86_64
./native/tests/openvino_version_coupling.sh
./native/tests/openvino_bundle_staging.sh windows-x86_64
```

Expected: PASS, PASS, PASS, then `SKIP: bundle not staged` for the Windows row on a Linux host.

- [ ] **Step 7: Commit**

```bash
git add native/build.sh native/tests/openvino_pin_selector.sh \
        native/tests/openvino_bundle_staging.sh native/tests/openvino_version_coupling.sh
git commit -m "feat(openvino): support the windows-x86_64 runtime bundle

The delegate has shipped in the Windows tarballs since the 1.4.1 pin and the
pin publishes a win_amd64 bundle; what was missing was a Java layer that could
extract six unversioned DLLs. It can now, so the platform joins the supported
set.

The staging test takes a platform argument and asserts a per-platform member
set -- including that the Windows BUILDINFO carries NO ov_abi, rather than
skipping the check there."
```

---

### Task 6: Run `openvinoTest` on Windows in CI

**Location:** Linux — this is a workflow YAML edit. The steps it adds run on the `windows-2022` runner, which is the acceptance gate; winbox is not.

**Files:**
- Modify: `.github/workflows/native-build-job.yml` — add steps to `build-executorch-shim-windows` after the CRT check (around line 233) and before the artifact upload

The tests go in the job that staged the bundle, matching the Linux precedent: `openvinoTest` runs inside `native-build-job.yml`'s `linux-x86_64` row rather than in `build-java-package`, whose download glob does not match the bundle's versioned filenames. The Windows row already uploads the whole staged tree, so no artifact plumbing changes.

- [ ] **Step 1: Add the JDK 17 toolchain and Gradle steps**

Insert after the "Assert the shim links the static CRT (MSVC)" step and before "Store executorch_djl shim":

```yaml
      # The build steps above bind JDK 8 deliberately, for the oldest supported jni.h. Gradle 9.6.1
      # cannot run on 8, so the test steps get their own toolchain -- the same split the Linux rows
      # use, where the container build runs under the image's Corretto 8 and Gradle runs after.
      - uses: actions/setup-java@v5
        with:
          distribution: zulu
          java-version: 17

      - uses: gradle/actions/setup-gradle@v6

      # Runs HERE, in the job that staged the bundle, for the same reason the Linux row does:
      # build-java-package's download glob does not match the bundle's library filenames, so the
      # bundle never reaches it and openvinoTest would silently skip there.
      - name: OpenVINO delegate tests
        shell: pwsh
        run: |
          .\gradlew.bat openvinoTest
          if ($LASTEXITCODE -ne 0) { throw "openvinoTest failed (exit $LASTEXITCODE)" }

      # The staging sync is done in PowerShell rather than inside bash: a `bash -c` string chaining
      # commands with && does not survive the PowerShell->native-exe quoting, so each Git-Bash call
      # below is one simple command with plain arguments -- the same shape as the CRT check above.
      - name: OpenVINO shell tests (staging, version coupling)
        shell: pwsh
        run: |
          New-Item -ItemType Directory -Force -Path build\native-staging\windows-x86_64 | Out-Null
          Copy-Item -Recurse -Force src\main\resources\native\windows-x86_64\* build\native-staging\windows-x86_64\
          $bash = "${env:ProgramFiles}\Git\bin\bash.exe"
          & $bash -c './native/tests/openvino_bundle_staging.sh windows-x86_64'
          if ($LASTEXITCODE -ne 0) { throw "openvino_bundle_staging failed (exit $LASTEXITCODE)" }
          & $bash -c './native/tests/openvino_version_coupling.sh windows-x86_64'
          if ($LASTEXITCODE -ne 0) { throw "openvino_version_coupling failed (exit $LASTEXITCODE)" }
```

`openvino_linkage.sh` is deliberately absent: it uses `nm` against an ELF `.so`. Windows delegate linkage is proven instead by the Catch2 case from Task 1 and by `openvinoTest` actually executing a delegated model.

`notices_staged.sh` is also absent: it reads a hardcoded `linux-x86_64` path. Parameterizing it is a notices concern rather than an OpenVINO one, so it belongs in its own change — note it in Task 10's follow-ups if you want it tracked.

- [ ] **Step 2: Update the artifact-upload comment**

The comment at the upload step says "Windows publishes no OpenVINO bundle today". Replace it:

```yaml
          # Whole staged tree, matching the Linux rows: the shim, its licences, and the OpenVINO
          # bundle that build.sh stages here.
```

- [ ] **Step 3: Verify the workflow parses**

```bash
python3 -c "import yaml,sys; yaml.safe_load(open('.github/workflows/native-build-job.yml')); print('OK')"
./native/tests/ci_workflow.sh
```

Expected: `OK` and a PASS. If `ci_workflow.sh` asserts something structural about the Windows job's step list, update it to match — that test exists to keep the two legs from drifting, so a failure here is it doing its job.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/native-build-job.yml
git commit -m "ci: run openvinoTest on windows-x86_64

In the job that stages the bundle, matching the Linux precedent -- the
downstream java job's download glob does not match the bundle's library
filenames, so openvinoTest would skip there. The Windows row already uploaded
the whole staged tree for this.

The job binds JDK 8 for jni.h, so the test steps bring their own JDK 17."
```

---

### Task 7: Verify the Gradle variant registers for Windows

**Location:** Linux. The variant is a Gradle metadata question, not a runtime one, so a synthetic
staged tree on this host answers it — no Windows machine is involved.

`nativePlatforms` already contains `windows-x86_64`, and both `nativeJar-<platform>-openvino`'s `onlyIf` and the `openvinoVariants` filter key on a staged `MANIFEST`. So the Windows variant should register itself with no Gradle edit. That is a prediction, and a wrong one fails at `generateMetadataFileForMavenPublication` — a release-time failure. Prove it locally instead.

**Files:**
- No source changes expected. If one proves necessary, it is in `build.gradle.kts:278-380`.

- [ ] **Step 1: Synthesize a staged Windows bundle on this Linux host**

The Gradle wiring only inspects file presence, so a structurally correct tree is enough to prove variant registration. The bytes do not need to be real DLLs.

```bash
D=build/native-staging/windows-x86_64
mkdir -p "$D/openvino/lib" "$D/openvino/licenses" "$D/licenses"
: > "$D/executorch_djl.dll"
: > "$D/licenses/LICENSE"
for f in openvino_c.dll openvino.dll openvino_intel_cpu_plugin.dll \
         openvino_ir_frontend.dll tbb12.dll tbbbind_2_5.dll; do : > "$D/openvino/lib/$f"; done
: > "$D/openvino/licenses/LICENSE"
printf 'openvino_version=2025.4.1\nplatform=windows-x86_64\n' > "$D/openvino/BUILDINFO"
printf 'openvino_version=2025.4.1\ntarball_sha256=deadbeef\ntarball_url=https://example.invalid/x.tar.gz\nlibs=openvino.dll openvino_c.dll openvino_intel_cpu_plugin.dll openvino_ir_frontend.dll tbb12.dll tbbbind_2_5.dll\nc_library=openvino_c.dll\n' > "$D/openvino/MANIFEST"
```

- [ ] **Step 2: Prove the variant and jar appear**

```bash
./gradlew nativeJar-windows-x86_64-openvino
./gradlew publishToMavenLocal
python3 - <<'PY'
import glob, sys
mods = glob.glob(str(__import__('pathlib').Path.home() / ".m2/repository/org/measly/djl-executorch-engine/*/*.module"))
text = open(sorted(mods)[-1]).read()
assert "windows-x86_64-openvino" in text, "windows openvino variant missing from module metadata"
print("OK:", sorted(mods)[-1])
PY
```

Expected: the jar task runs (not `SKIPPED`), `publishToMavenLocal` succeeds, and the script prints `OK`. A `FileNotFoundException` from `generateMetadataFileForMavenPublication` is the failure this task exists to catch — it means the variant registered without its artifact, and the fix is in the `onlyIf`/filter pair in `build.gradle.kts`.

- [ ] **Step 3: Clean up the synthetic tree**

```bash
rm -rf build/native-staging/windows-x86_64
rm -rf ~/.m2/repository/org/measly/djl-executorch-engine
```

The synthetic tree must not survive into any later task — a jar built from empty files is exactly the thing the release gates exist to prevent.

- [ ] **Step 4: Record the result**

No commit if no change was needed; note the outcome in the task notes. If `build.gradle.kts` did need an edit, commit it with a message stating which prediction was wrong.

---

### Task 8: Documentation

**Location:** Linux.

**Files:**
- Modify: `CLAUDE.md` (the OpenVINO bullet's platform statements and the supported-set paragraph)
- Modify: `docs/openvino-version-bump.md` (items 1 and 3, and the "must NOT change" list)

- [ ] **Step 1: Update the OpenVINO bullet in `CLAUDE.md`**

Two statements are now false. The bundle sentence says the bundle is published per platform and the engine stages only `linux-x86_64`; and the supported-set paragraph says "The pin publishes a `windows-x86_64` bundle that this engine declines, because `OpenVinoRuntime` cannot yet extract it."

Replace that paragraph with:

```markdown
  Which platforms get a bundle staged is an **engine-side decision**, not a mirror of what the pin
  publishes: `ET_OPENVINO_SUPPORTED_PLATFORMS` in `native/build.sh` (`linux-x86_64 windows-x86_64`)
  is the list, and the pin's per-platform rows are consulted only for platforms on it. A platform
  joins that list once `OpenVinoRuntime` can extract its bundle and a test proves it. The lookup
  keys on the platform identity, never the pin row — the `windows-x86_64-static` row is an alias
  holding a CMake variable reference that shell cannot dereference.

  The two platforms' bundles differ in shape, and the engine reads rather than reconstructs: the
  staged `MANIFEST` carries `libs` (the filenames) and `c_library` (the one to point
  `OPENVINO_LIB_PATH` at). Linux ships seven ABI-versioned libraries and an `ov_abi` key in
  `BUILDINFO`; Windows ships six unversioned DLLs, no `ov_abi` key at all, and no separate hwloc
  (it is folded into `tbbbind_2_5.dll`). Nothing in Java derives a filename from a version.
```

- [ ] **Step 2: Update `docs/openvino-version-bump.md`**

Item 1 still names the deprecated `ET_RUNTIME_OPENVINO_{VERSION,PLATFORM,URL,SHA256}` quartet, which sub-project A stopped reading. Replace its last sentence with:

```markdown
   This carries `ET_RUNTIME_OPENVINO_VERSION` and the per-platform
   `ET_RUNTIME_OPENVINO_URL_<platform>` / `ET_RUNTIME_OPENVINO_SHA256_<platform>` rows.
```

Item 3 tells the reader to update `OpenVinoRuntime.LIBS`, which no longer exists. Replace item 3 entirely:

```markdown
3. **`native/tests/openvino_bundle_staging.sh`** — only if the bundle's library set changed. It
   holds the expected member set per platform, and it is the only place that enumeration lives:
   `OpenVinoRuntime` reads whatever the staged `MANIFEST` declares, so nothing in Java needs
   editing. A count mismatch from this test is the signal to come here.
```

Add to the "What must NOT change" list:

```markdown
- **No Java file lists the bundle's libraries.** `MANIFEST`'s `libs` and `c_library` are generated
  from the staged tree by `native/build.sh`. If you find yourself adding a filename to a `List<String>`
  in `OpenVinoRuntime`, something has regressed.
```

Add the Windows row to the verification block:

```bash
./native/tests/openvino_bundle_staging.sh windows-x86_64   # on winbox, after staging there
```

- [ ] **Step 3: Verify the docs tests pass**

```bash
./native/tests/docs_present.sh
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/openvino-version-bump.md
git commit -m "docs: OpenVINO ships on windows-x86_64

Records the two bundle shapes and that no Java file lists libraries any more
-- the version-bump checklist pointed at OpenVinoRuntime.LIBS, which no
longer exists, and at the pin's deprecated singular vars, which are no longer
read."
```

---

### Task 9: Windows end-to-end verification on winbox

**Location:** **winbox**, every step. No source changes.

This verifies the capability on a real Windows host. It is *not* sign-off: winbox runs VS 18
Community against the runner's VS 17 Enterprise, so the `windows-2022` job from Task 6 is the
acceptance gate. What this catches that CI cannot is anything needing a human to look at it.

- [ ] **Step 1 (winbox): Get the branch onto winbox**

Push the branch from the Linux box, then bind `$repo` and update the checkout — see "Working on
winbox" for both halves and the `git bundle` fallback. `$repo` must stay bound for the rest of this
task, along with `$bash` and `$run`; every block below passes `$repo` explicitly rather than relying
on a working directory.

Bind the session variables from "Working on winbox" and `scp` the helper over, then update the
checkout. `git -C` takes the repo explicitly, so this block needs no working directory either:

```powershell
$repo = "<the djl-executorch-engine checkout path on winbox>"
$bash = "${env:ProgramFiles}\Git\bin\bash.exe"
$run  = "$env:USERPROFILE/winbox-run.sh"
git -C $repo fetch origin
git -C $repo checkout feature/openvino-windows
git -C $repo reset --hard origin/feature/openvino-windows
Remove-Item -Recurse -Force "$repo\native\build", "$repo\native\asan" -ErrorAction SilentlyContinue
```

A stale CMake cache is worth deleting rather than debugging: one configured for a different source
root or ExecuTorch version produces failures that read like compiler bugs. Drive the session in
short chunks, redirecting stdin on the local `ssh` call rather than inside the remote command.

- [ ] **Step 2 (winbox): Activate the MSVC dev shell**

`-SkipAutomaticLocation` is required, not optional: without it `Launch-VsDevShell.ps1` changes the
working directory to the Visual Studio default, and every relative path afterwards resolves against
the wrong root. CI passes it for the same reason.

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -SkipAutomaticLocation
```

It mutates *this* PowerShell process's environment in place, so the Git-Bash children launched by
later steps inherit the MSVC toolchain. That is also why `-c`, never `-lc`: a login shell re-sources
the profile and drops it.

- [ ] **Step 3 (winbox): Build and stage**

```powershell
& $bash $run $repo ./native/build.sh
```

Expected: `executorch_djl.dll` staged, and — for the first time on this platform — `OpenVINO bundle staged:` naming `src/main/resources/native/windows-x86_64/openvino`. Confirm the six DLLs and the MANIFEST keys:

```powershell
Get-ChildItem "$repo\src\main\resources\native\windows-x86_64\openvino\lib" | Select-Object -ExpandProperty Name
Get-Content "$repo\src\main\resources\native\windows-x86_64\openvino\MANIFEST"
```

Done in PowerShell rather than Git-Bash: a `bash -c` string chaining two commands with `&&` is
exactly the quoting hazard described above.

- [ ] **Step 4 (winbox): Run the CRT check and the shell tests**

```powershell
& $bash $run $repo ./native/tests/check_windows_crt.sh native/build src/main/resources/native/windows-x86_64/executorch_djl.dll
New-Item -ItemType Directory -Force -Path "$repo\build\native-staging\windows-x86_64" | Out-Null
Copy-Item -Recurse -Force "$repo\src\main\resources\native\windows-x86_64\*" "$repo\build\native-staging\windows-x86_64\"
& $bash $run $repo ./native/tests/openvino_bundle_staging.sh windows-x86_64
& $bash $run $repo ./native/tests/openvino_version_coupling.sh windows-x86_64
```

Expected: PASS from each. One command with plain arguments per call — no `&&` chain, and no reliance
on what working directory PowerShell handed the child.

- [ ] **Step 5 (winbox): Run the OpenVINO JVM tests**

```powershell
& "$repo\gradlew.bat" -p $repo --no-daemon --console=plain openvinoTest
```

Expected: green — `OpenVinoRuntimeTest` (extraction into `%LOCALAPPDATA%\executorch-djl\openvino\<sha>\`), `OpenVinoConcurrentExtractionTest` (the atomic-rename publication, which matters most on the platform that refuses to delete a loaded library), and `OpenVinoModelIT` (parity at `atol=1e-2`).

- [ ] **Step 6 (winbox): Run the full Windows suite**

```powershell
& "$repo\gradlew.bat" -p $repo --no-daemon --console=plain test
```

Expected: green. This is the regression check that Task 3's extractor rewrite did not disturb anything else.

- [ ] **Step 7 (winbox): Clean up and record the results**

Remove the two helper scripts from the checkout, so a later `git status` there is clean:

```powershell
Remove-Item "$repo\winbox-run.sh", "$repo\ov-smoke-stage.sh" -ErrorAction SilentlyContinue
```

No commit. Note each outcome, and record which precision `openVinoInferencePrecision()` reported on winbox — it is useful context for reading any future parity failure.

---

### Task 10: Linux regression, follow-ups, and the PR

**Location:** Linux.

- [ ] **Step 1: Full Linux gate**

```bash
rm -rf native/build
./native/local_build_wrapper.sh
for t in native/tests/*.sh; do
  case "$t" in *check_windows_crt.sh) continue ;; esac
  echo "== $t"; bash "$t" || echo "FAILED: $t"
done
./gradlew test
./gradlew openvinoTest
./native/local_build_wrapper.sh native/build_qa.sh
```

Expected: no `FAILED:` lines and all suites green. Tasks 2, 3 and 4 all changed shared code paths, so this proves Linux is unaffected.

- [ ] **Step 2: File the producer documentation issue**

```bash
gh issue create --repo measly-java-learning/executorch-runtime-dist \
  --title "openvino-jni-consumer.md still says Linux x86_64 only" \
  --body 'The doc was edited in the win_amd64 publication PR, but only its selector section. As published at `v1.4.1-2` it still opens "Linux `x86_64` only — that is the only platform where this delegate exists", and its library table, `setenv` recipe and checklist are all `.so`-shaped.

The Windows shape is in `docs/handover-to-engine.md` C10 instead — six unversioned DLLs, no symlink, no hwloc entry, `OPENVINO_LIB_PATH` naming `lib/openvino_c.dll` — so the two documents disagree about which platforms the delegate exists on.

Reported from the DJL ExecuTorch engine, which has now shipped the Windows bundle and had to take the library list from C10 rather than from the recipe doc.'
```

- [ ] **Step 3: File the winbox documentation-consolidation issue**

```bash
gh issue create --title "winbox mechanics are duplicated and contradictory across old plan docs" --body 'Driving the Windows iteration host is documented nowhere current. It is scattered across completed plan documents under `docs/superpowers/plans/`, which read as instructions, are what a grep surfaces first, and now contradict each other:

- `2026-07-18-windows-static-crt.md:160` — "**Default remote shell is `cmd`**, not PowerShell or bash", plus `powershell -NoProfile -EncodedCommand` recipes. **False:** sshd'"'"'s `DefaultShell` is pwsh 7.x. `2026-07-15-windows-builds.md` carries the same `-EncodedCommand` pattern in three places.
- `2026-07-18-windows-static-crt.md:321` — winbox "is provisioned as a *native build* host ... not a JVM toolchain", so a Gradle run may be impossible there. **Superseded** by `2026-08-09-production-observability.md:444`, which records Zulu 17.0.19 confirmed present.
- `2026-07-15-windows-builds.md:1025` uses `rsync` with an `scp` fallback; there is no rsync on winbox.
- The one thing that IS accurate and easy to miss: `scp` needs absolute forward-slash targets, because a bare relative path fails on this sftp server (`2026-07-18-windows-static-crt.md:169`).

Each of these cost real time in the OpenVINO Windows work, and two of them produced commands that could not have run.

**Proposal, not blanket deletion.** These documents are worth keeping for their reasoning, per the docs convention in CLAUDE.md. What is missing is one piece of *current guidance* they can defer to, and a pointer to it:

1. Add `docs/windows-iteration-host.md` as the single current reference: shell (pwsh 7, no `<` redirection), VS activation with `-SkipAutomaticLocation`, the Git-Bash handoff and its quoting limits, scp path rules, the absence of rsync, what toolchains are present (VS, Git-Bash, JDK 17), and that winbox is an iteration host while the `windows-2022` runner is the acceptance gate.
2. Add a routing line to `CLAUDE.md`. This is the highest-leverage item and the cheapest: CLAUDE.md is loaded every session, so it is what actually prevents the next occurrence. Roughly: `docs/superpowers/plans/` and `specs/` record what was true when written and are not current guidance; for Windows host mechanics see `docs/windows-iteration-host.md`.
3. Add the new doc to `docs/README.md`.
4. Put a one-line superseded banner at the top of each plan above pointing at it, rather than editing their bodies — they are point-in-time records and rewriting them would falsify what was known then.
5. Consider whether `docs/research/handover-windows-static-cxx17-findings.md` should also carry one.

The machine-specific parts (hostname, user, key path) stay out of the repo, in `windows-jni-handoff.md`, as today.

**Considered and rejected: hiding the stale files from Claude.** There is no `.claudignore` in Claude Code; the equivalent is a `permissions.deny` entry such as `Read(./docs/superpowers/plans/2026-07-18-windows-static-crt.md)` in `.claude/settings.json`. It is the wrong tool here for three reasons, the first decisive:

- Staleness is **per claim, not per file**. The document carrying the false `cmd` claim is the only place recording the true scp rule, and hiding it would have suppressed the fix along with the bug.
- A denied read fails **silently** — the reader cannot know the document exists, so it cannot route to a better one. A superseded banner fails loudly and points forward.
- The deny list would go stale exactly as the docs did, and less visibly, so it adds a second thing to maintain instead of maintaining the first. It also does nothing for humans, who grep these files too and are misled identically.

A deny rule is only right for a document that is entirely dead, and then deleting it is better, since git keeps the history.'
```

- [ ] **Step 4: File the general Windows JVM CI issue**

```bash
gh issue create --title "No Windows JVM test job in CI beyond openvinoTest" --body 'The OpenVINO Windows work added `gradlew.bat openvinoTest` to `build-executorch-shim-windows`. The wider gap it exposed is that no other Windows JVM test runs in CI at all: `native-build-job.yml`'"'"'s Windows row builds and uploads the DLL, and every other Gradle suite runs on `ubuntu-latest`. `gradlew.bat test` remains a manual winbox step.

That matters because Catch2 on Windows links only the JNIEnv-free core, so nothing in CI exercises the JNI signatures on that platform — the failure mode #16/#17 were about.

The steps now in the Windows job are the template; the question is runner-minute budget, which is why this is separate from the OpenVINO change.'
```

- [ ] **Step 5: Push and open the PR**

Write the body to a file — it carries evidence from Tasks 1, 7, 9 and 10 — covering: that the delegate already linked on Windows and only the Java extraction blocked it; the `MANIFEST`-declares-its-contents move and what it deleted (`LIBS`, the ABI-suffix construction, the unversioned-plugin fallback); the two bundle shapes; the Task 1 flat-directory proof on winbox with its result; the Gradle variant verification from Task 7; the Windows and Linux gate results; and links to the three issues.

```bash
git push -u origin feature/openvino-windows
gh pr create --title "Ship the OpenVINO runtime bundle for windows-x86_64" --body-file /tmp/ovwin/pr-body.md
```

- [ ] **Step 6: Close issue #53 on merge and hand off**

Reference `Closes #53` in the PR body, then use `superpowers:finishing-a-development-branch` to decide how this integrates.

---

## Notes for the executor

- **Task 1 is the go/no-go, and it runs on winbox before any Java is touched.** If a flat directory does not resolve OpenVINO's plugins on Windows, the spec's §3 and §4 are both void and the fix belongs in the producer's bundle layout. Report it rather than working around it.
- **The staging test must never read `MANIFEST`.** It is the independent enumeration; a manifest generated from a truncated bundle would describe that truncation accurately and pass.
- **Do not tighten `atol=1e-2`.** Both `f32` and `bf16` are correct answers from `openVinoInferencePrecision()`, and a tighter bound asserts which machine CI allocated.
- **The synthetic tree in Task 7 is throwaway.** Delete it before Task 9, or a later gate will inspect a jar built from empty files.
