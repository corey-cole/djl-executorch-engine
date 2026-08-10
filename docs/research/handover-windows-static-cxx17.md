> **Historical record — not current guidance.**
> Written 2026-07. Kept for the reasoning it captures; details may have been superseded by later
> work. For current guidance see [docs/README.md](../README.md).

# Engine Work Order — Windows static CRT + C++17 propagation check

> **Who this is for:** the agent working in the **DJL ExecuTorch engine** (the repo that builds
> `libexecutorch_djl.so` / the JNI DLL). You are expected to know that repo; you are **not** expected
> to know the producer repo, `measly-java-learning/executorch-runtime-dist`.
>
> **Scope:** two independent work streams. Stream A is a concrete adoption task. Stream B is an
> investigation that may end in "nothing to change" — that is a valid outcome, provided you can say
> *why*.
>
> **Background (only if you need it):** `docs/handover-to-engine.md` in the producer repo is the
> full cold-start hand-off with the frozen C1–C9 contract. Its §1 names an old release tag; ignore
> that and resolve the latest release as described below. Everything you strictly need is here.

---

## 0. First, resolve the release

Do **not** hardcode a release tag, URL, or SHA256 anywhere in the engine. Resolve the latest release
of the producer repo and take its `EtRuntimePin.cmake` asset — that file is the single source of
truth for URLs and hashes:

```bash
gh release download --repo measly-java-learning/executorch-runtime-dist \
  --pattern 'EtRuntimePin.cmake' --dir native/cmake/
```

It defines, per row, `ET_RUNTIME_URL_<variant>_<platform>` and
`ET_RUNTIME_SHA256_<variant>_<platform>`. The rows you care about:

| Platform token | CRT | Intended consumer |
|---|---|---|
| `windows-x86_64` | `/MD` (dynamic) | CPython extensions — must match CPython's own CRT |
| `windows-x86_64-static` | `/MT` (static) | **the JNI DLL — use this one** |

Windows ships the `logging` variant only. Linux ships all three variants on both `x86_64` and
`aarch64`; nothing about Linux changes in this work order.

---

# Stream A — Adopt `windows-x86_64-static` for the JNI DLL

## Why

The `/MT` artifact folds the C runtime into your DLL, so **end users need no VC++ redistributable**.
That is the entire point: Windows consumers of the engine are developers on potentially locked-down
workstations who may not be able to run an installer. The `/MD` row remains published for the Python
side and is not the right choice here.

## A1. Point Windows at the `-static` row

Wherever the engine maps a detected platform to a pin row, Windows must select
`windows-x86_64-static`. The extracted tarball's top-level directory carries the same token:

```
executorch-runtime-<etver>-logging-windows-x86_64-static/
```

so any path construction that interpolates the platform token keeps working — but check for a
hardcoded `windows-x86_64` string, which will now silently resolve to the wrong (dynamic) row.

## A2. Compile the JNI target `/MT` to match

```cmake
set_property(TARGET <your_jni_target> PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
```

This requires CMake policy `CMP0091` to be `NEW` (automatic with `cmake_minimum_required(VERSION 3.15)`
or later). Apply it to **every** target in the engine that gets linked into the final DLL, not only
the top-level one — a static library compiled `/MD` and linked into a `/MT` DLL reintroduces exactly
the problem you are removing.

## A3. ⚠ Do not expect the linker to catch a mismatch

This was **measured, not assumed**: a `/MD` consumer linked a `/MT` ExecuTorch artifact with no error
and **no `LNK4098` warning**. Do not treat a clean link as evidence you got this right.

The failure mode is at runtime, not link time: two CRTs, two heaps, and corruption when an allocation
crosses the boundary. Since JNI's ABI is pure C and never hands CRT-owned resources (`FILE*`,
allocations, locale) across the JVM boundary, `/MT` is safe *for the JNI DLL specifically* — but only
if the whole DLL is consistently `/MT`.

## A4. Verify the CRT empirically

Confirm what actually got baked into your objects rather than trusting the build files:

```bash
dumpbin -nologo -directives <your>.lib | grep -i defaultlib
```

Expect `LIBCMT` / `LIBCPMT` (static). Seeing `MSVCRT` / `MSVCPRT` means something is still `/MD`.

Two traps worth knowing, both of which cost the producer repo real debugging time:

- **Use the dash form `-nologo -directives`, not `/nologo`.** Under MSYS / Git Bash, a leading `/`
  is path-converted into something like `C:\Program Files\Git\nologo`, and `dumpbin` fails. It can
  fail on *every* library while your check still reports success.
- **Assert on the presence of the expected marker**, never on the absence of the wrong one. An
  absence check passes when the tool fails to run at all.

The producer repo's `scripts/check-windows-crt.sh` is a working implementation of this scan if you
want a reference.

## A5. Prove the redistributable is actually unnecessary

The user-facing claim is "no VC++ redist required," so test that claim directly: load the JNI DLL on
a Windows machine (or container) that has **never** had a VC++ redistributable or Visual Studio
installed. A dev box will not tell you anything — it already has the runtime.

## Stream A — done when

- [ ] Windows resolves the `windows-x86_64-static` pin row; no hardcoded `windows-x86_64` remains
- [ ] Every target linked into the DLL is `MSVC_RUNTIME_LIBRARY "MultiThreaded"`
- [ ] `dumpbin -nologo -directives` shows `LIBCMT`/`LIBCPMT` and no `MSVCRT`/`MSVCPRT`
- [ ] The DLL loads and runs inference on a clean Windows image with no redistributable
- [ ] Existing Linux paths are untouched and still green

---

# Stream B — C++17 propagation investigation

## The finding

ExecuTorch's public headers require C++17 and enforce it with a hard `#error`
(`runtime/platform/compiler.h`). But the installed CMake package **does not export that requirement**
— `INTERFACE_COMPILE_FEATURES` is absent from every exported target (verified on both Linux and
Windows builds of v1.3.1). ET sets `CMAKE_CXX_STANDARD 17` for its own build, which is a build-time
variable, not a usage requirement.

So `find_package(ExecuTorch REQUIRED)` gives you include dirs, compile definitions, compile options
and link options — **but nothing about the language standard.** Every consumer must state C++17
itself.

This is invisible on Linux (GCC defaults to `gnu++17`) and fails hard on MSVC (defaults to C++14):

```
compiler.h(36): fatal error C1189: #error: "You need C++17 to compile ExecuTorch"
```

## Why you are being asked to look, given nothing is broken

**The guard is `#error`, not undefined behaviour.** A build either fails loudly or C++17 was set. So
the fact that the engine builds on Windows today is *proof* it is already getting C++17 from
somewhere. There is no silent-corruption variant of this.

The question is therefore **not** "is it broken" — it is **"where is C++17 coming from, and how
fragile is that source?"** An implicit or third-party-supplied standard can vanish under an unrelated
dependency bump, and it will surface as a confusing MSVC-only failure in a Windows CI job.

Stream A makes this timely: editing the engine's CMake to switch pin rows is exactly the kind of
change that disturbs an unstated requirement.

## B1. Find the source of the standard

```bash
grep -rn "CXX_STANDARD\|cxx_std_\|std:c++\|std=c++" \
  --include=CMakeLists.txt --include="*.cmake" .
```

A **zero-hit result on a project that builds on MSVC is the interesting case** — something implicit
is supplying it and you do not control it.

## B2. If it is `CMAKE_CXX_STANDARD`, check it is actually required

```bash
grep -rn "CMAKE_CXX_STANDARD_REQUIRED" --include=CMakeLists.txt --include="*.cmake" .
```

`set(CMAKE_CXX_STANDARD 17)` **without** `set(CMAKE_CXX_STANDARD_REQUIRED ON)` is only a *preference*
— CMake will silently decay to an older standard if it believes the compiler cannot manage 17, and
you would then hit the `#error` with no obvious cause.

## B3. Check the scope

A directory-scope `set(CMAKE_CXX_STANDARD 17)` applies only to targets created **after** it, in that
directory and below. A target defined in a subdirectory added earlier, or in a sibling tree, does not
get it. Prefer the per-target form, which cannot drift:

```cmake
target_compile_features(<target> PRIVATE cxx_std_17)
```

## B4. Confirm every TU that includes ET headers is covered

The standard must apply to **the target that compiles the ET-including translation unit**, not merely
to some target in the project. If the JNI shim and its ET-facing wrapper live in different targets,
check both.

## B5. Verify on MSVC specifically

Linux tells you nothing here — `gnu++17` masks the entire issue. Run this against the **Windows**
build:

```bash
cmake --build <build-dir> -- -v 2>&1 | grep -o '/std:c++[0-9a-z]*' | sort -u
```

No `/std:` in the output means MSVC is on its C++14 default and you are relying on nothing.

Note the ET guard keys on `_MSVC_LANG`, so `/std:c++17` alone satisfies it — `/Zc:__cplusplus` is
**not** required.

## Stream B — done when

- [ ] You can state, in one sentence, where the engine's C++17 requirement comes from on Windows
- [ ] If that source is implicit or third-party, the engine declares C++17 explicitly itself
- [ ] Every target compiling an ET-including TU is covered, verified via B5 on a Windows build
- [ ] A line in the engine's README/build docs noting that `find_package(ExecuTorch)` does **not**
      supply the language standard

## If it does resurface

The fingerprint is always Windows/MSVC-only. Grep a failing build log for `C1189` or `_MSVC_LANG`.
Likeliest triggers: an unrelated cleanup deleting a `set(CMAKE_CXX_STANDARD 17)` line, or a build
backend upgrade dropping a standard it used to supply implicitly.

Fixing this upstream in ExecuTorch is **the producer repo's open item, not yours** — do not file an
issue against `pytorch/executorch` from the engine repo. Report what you find back to the producer
repo instead; it holds a drafted proposal awaiting the next ET release.

---

## Reporting back

Both streams are worth a short written result. In particular, if Stream B ends in "nothing to
change," say where the standard comes from and why that source is trustworthy — that conclusion is
only useful with its evidence attached.
