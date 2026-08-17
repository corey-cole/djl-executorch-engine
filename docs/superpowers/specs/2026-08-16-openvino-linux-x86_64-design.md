# OpenVINO delegate support

**Date:** 2026-08-16
**Status:** approved design
**Follows:** the v1.3.1-10 pin bump and the XNNPACK workspace metric, in that order

**Correction (verified after implementation):** this document was written on the premise, taken
from upstream's consumer doc and the sibling project's spec, that the OpenVINO delegate ships on
`linux-x86_64` only. That is false. `libopenvino_backend.a` is in the **`linux-aarch64` tarball
too** — confirmed by listing the shipped archive, and by CI logging `OpenVINO delegate: linked` on
that leg. What is `linux-x86_64`-only is the OpenVINO **runtime bundle**
(`ET_RUNTIME_OPENVINO_PLATFORM`). Windows is the only platform with no delegate. Read every
"linux-x86_64 only" below as applying to the runtime bundle, not the delegate. The practical
consequence is a third state this design did not anticipate — delegate linked, no runtime available
— which needs its own message, because re-exporting would be the wrong advice there.

Makes the OpenVINO delegate — compiled into the Linux runtime tarballs as
`lib/libopenvino_backend.a` — a supported feature of this engine: an opt-in qualified jar carrying
the OpenVINO runtime, automatic `OPENVINO_LIB_PATH` resolution from JNI, a committed fixture, and CI
that executes a delegated model rather than merely proving the archive linked.

Nothing here touches export. The partitioner and the quantizer stay out of scope permanently.

## The constraint everything else is shaped around

The delegate resolves the OpenVINO C API with `dlopen` at first use, reading `OPENVINO_LIB_PATH`.
That load happens **once**, under `std::call_once`, **with no retry**. A first attempt that fails
leaves the process broken until it restarts.

So the organising principle is: **every failure we can detect must be raised before ExecuTorch is
entered.** A misconfiguration that reaches `call_once` costs the user their JVM; the same
misconfiguration caught one frame earlier costs them a stack trace they can act on.

This engine's version of the problem is strictly harder than the Python consumer's. A Python
consumer resolves the path inside an installed `openvino` wheel and is done. We must first get
~69 MB of shared libraries onto disk, and we cannot fall back on `LD_LIBRARY_PATH` at all: glibc's
loader reads it once at process start, `System.getenv` is read-only, and `ProcessBuilder` affects
only children. Even `setenv("LD_LIBRARY_PATH", …)` from JNI is too late to influence a later
`dlopen`. `OPENVINO_LIB_PATH` is read at `dlopen` time, which makes it the **only** mechanism
available to a JVM.

## Architecture

Four touch points, no new components.

```
EtModel.load ──uses_backend?──▶ OpenVinoRuntime.ensureReady()
     │                             ├─ extract bundle (content-addressed, atomic dir publish)
     │                             └─ EtNative.setOpenVinoLibPath(dir)  → setenv(), once
     ▼
EtRuntime ctor (C++) ──guard──▶ throws if backend unregistered or path unusable
     ▼
OpenvinoBackend::init ──call_once──▶ dlopen   ← never reached misconfigured
```

**Detection uses metadata, not a byte scan.** `MethodMeta::uses_backend("OpenvinoBackend")`
(`runtime/executor/method_meta.h:257`) answers exactly the right question, and answers it before the
method loads. A new `EtNative.methodUsesBackend(handle, backend)` wraps it; both the Java resolver
and the C++ guard read through it. Byte-scanning the `.pte` for `b"OpenvinoBackend"` also works and
upstream's emitter does it as a post-export check, but it is a heuristic over a flatbuffer and it
forces reading a whole model file ExecuTorch is about to read again. `uses_backend` is exact.

**Detection is required regardless of extraction timing**, purely to produce a decent error. That
is what makes lazy extraction free: it rides on a signal we already need.

### Where each half of the detection runs, and why they differ

`EtNative.loadModule` constructs an `EtRuntime`, whose constructor calls `Module::load_forward()`
unconditionally — and that call *is* delegate init. So nothing after `loadModule` can detect
anything in time. The two halves therefore sit in different places:

- **The guard is C++, inside the `EtRuntime` constructor, between `Module` construction and
  `load_forward()`.** At that point the program is loaded and `method_meta` is available, so
  `uses_backend` costs nothing extra, and every error case — backend unlinked, `OPENVINO_LIB_PATH`
  unset or not a file — is raised before delegate init. This is where all four errors come from.
- **The Java probe is a separate `EtNative.pteUsesBackend(path, backend)` call before
  `loadModule`,** because only Java can extract the bundle, and only Java knows the platform and
  whether a bundle is on the classpath — which is what lets it say something the C++ guard cannot.

**Revised after measuring (supersedes this design's original "conditional probe").** The probe was
originally skipped when no bundle was present, to avoid "one extra `.pte` open" on platforms that
could not act on it. Measured on a 4-core x86_64 host, that caution was unwarranted: the probe costs
**6.5–10 µs and is essentially flat in model size** (1.6 KB → 2.7 MB), because `method_meta` reads
program metadata and never the weights. Against model load it is 35% for a toy 1.6 KB model and
**1.5% for a 2.7 MB one**, paid once per load rather than per inference — so it shrinks to noise
exactly as models get realistic.

The probe is therefore **unconditional**, which buys a correct error where the cheap version had to
stay silent and let a less-informed layer report. It still returns immediately once configured, and
still does nothing for a model that does not use the backend.

**The C++ guard duplicates the Java check deliberately.** `EtNative` is public and bypasses
`EtModel`, and our own tests use it directly. Without the guard a direct `EtNative` caller who
misconfigures OpenVINO burns the process's `call_once` and cannot recover without a restart.

### Rejected alternatives

**Eager extraction on classpath presence.** Treating "the OpenVINO jar is present" as the opt-in and
extracting at native-library load is simpler to order, but it pays 69 MB of disk and extraction time
for a user who added the jar for one model among many — and it still needs detection for the error
path, so it is strictly more work for the same result.

**An explicit `EtEngine.enableOpenVino()`.** Forgetting it produces exactly the unrecoverable
`call_once` failure this design exists to prevent, reported from deep inside ExecuTorch rather than
from the call that was skipped.

## Packaging

The OpenVINO runtime ships in a **separate opt-in jar variant**, not in the standard platform jar.

`build.gradle.kts` already publishes each platform's natives as a real GMM variant with a
per-platform capability (`org.measly:djl-executorch-engine-<platform>`), which keeps the variants out
of default resolution. An OpenVINO bundle becomes another variant alongside them, with its own
capability, that a consumer adds deliberately.

The alternative — folding it into the standard `linux-x86_64` jar — would cost every consumer on
that platform ~21 MB compressed and ~69 MB extracted for a delegate most will never load. The
delegate is optional by nature; the packaging should say so.

**What the bundle contains** is upstream's `openvino-runtime-<ovver>-<platform>.tar.gz`, pinned by
`ET_RUNTIME_OPENVINO_URL` / `ET_RUNTIME_OPENVINO_SHA256`: seven libraries plus one symlink in a flat
`lib/`, about 69 MB on disk. The GPU/NPU plugins and the ONNX/TF/PyTorch/Paddle/JAX frontends are
already excluded upstream. `libopenvino_ir_frontend` is **not** one of the excludable frontends
despite the name — the blob a `.pte` carries is OpenVINO IR and importing it needs that library.
Without it the runtime loads, reports a CPU device, and then fails every model at load with
`failed to import model for device 'CPU' (status=-1)`.

Ship the bundle's `licenses/` too: Apache 2.0 plus the hwloc BSD-3-Clause notice. The existing
`nativeJar-<platform>` tasks already have the pattern for staging notices into
`META-INF/licenses/`, and already fail the build rather than ship a binary without them.

**Linking is capability-driven, never platform-driven.** In CMake:

```cmake
if(TARGET openvino_backend)
  target_link_libraries(executorch_djl PRIVATE openvino_backend)
endif()
```

Keyed on the target existing rather than on `ET_PLATFORM`, so a platform that later gains the
delegate works with no edit. For the same reason this must **not** join the post-link
`assert_xnnpack_registered.cmake` guard's required set: that guard exists to catch the XNNPACK
registration being GC'd out of the `.so`, and a target that legitimately does not exist on most
platforms would turn a real guard into a platform conditional. If OpenVINO needs an equivalent
registration assertion, it gets its own, conditional on the same `TARGET` test.

`libopenvino_backend.a` is static and `dlopen`s its dependency, so linking it adds no `DT_NEEDED`
and drags in no OpenVINO shared object at link time.

## Extraction and caching

All bundle libraries must land in **one flat directory**. Every one carries `RPATH=$ORIGIN`, which
is what resolves the entire dependency graph with no `LD_LIBRARY_PATH`, no `ldconfig`, and no system
install. Splitting them across directories breaks that.

**Cache key: the upstream bundle SHA256**, carried in a `MANIFEST` resource at
`/native/<platform>/openvino/MANIFEST` inside the jar, mirroring `ET_RUNTIME_OPENVINO_SHA256` from
the pin. Flat `key=value`, so the version-coupling check below can read it from shell without a
parser — the same reason the fixture manifest uses that format.

This departs from `LibUtils` deliberately. `LibUtils` derives its key by hashing the resource, so key
and content are the same fact — but it pays a full read of the library on *every* JVM start, hit or
miss. That is tolerable at 12 MB and not at 69 MB, on a path that runs at model load. Taking the key
from the manifest makes a cache hit cost zero reads. The tradeoff accepted in exchange: a corrupted
jar yields a directory keyed by the right hash holding wrong bytes. That is the jar's integrity
problem, one layer down, and the same trust already extended to every other class on the classpath.

**Publish by atomic directory rename.** Extract into a sibling staging directory, then `ATOMIC_MOVE`
the directory into `<cacheRoot>/openvino/<sha256>/`. A loser in a race adopts the winner's directory
exactly as `LibUtils.extract` already does for the shim — the path is content-addressed, so the
winner's bytes are ours byte-for-byte. This also disposes of partial extraction without a marker
file: the directory only ever appears complete.

`LibUtils.cacheRoot()` is reused unchanged, so `XDG_CACHE_HOME` and `LOCALAPPDATA` are already
honoured. The root is stable by construction, which satisfies the consumer doc's warning that a temp
directory cleaned between the `setenv` and the first inference fails exactly as if the variable were
never set.

**The invariant that keeps this portable: never load out of the staging directory.** Publish first,
load second. On a platform that refuses to delete a loaded library, the loser's cleanup must be able
to delete every file it wrote, which holds only if nothing was ever loaded from there.

## Error taxonomy

Four cases, every one raised before ExecuTorch is entered.

| Condition | The message names |
|---|---|
| Model uses `OpenvinoBackend`, delegate not linked | the running platform, that the delegate ships only where upstream builds it, and re-export as the fix |
| Delegate linked, bundle jar absent | the missing artifact and its Gradle coordinates |
| Caller-set `OPENVINO_LIB_PATH` is not a file | the offending value, and that it wants the **full path to the library file, not a directory** |
| Caller-set `OPENVINO_LIB_PATH` is a file | nothing — it wins, untouched |

The third row earns its place. Upstream lists "set to a directory" as the top mistake precisely
because the error you otherwise get mentions `LD_LIBRARY_PATH`, which reads like it wants a
directory. It does not.

The first row earns its place too: without it, an OpenVINO `.pte` on a platform lacking the delegate
falls through the existing load-failure path and reports a corrupt or version-mismatched model.
That actively misdirects — the `.pte` is fine, the platform cannot run it.

Honouring a caller-set path is an explicit "is it already set?" check, never a set-if-absent
idiom whose eager default would try to resolve the bundle — and fail when it is absent — even when
the variable is already correct.

These throw DJL's `EngineException`, unchecked and engine-native, rather than a new type.

## Precision accessor

`EtEngine.openVinoInferencePrecision()` returns the numeric type OpenVINO will use for CPU inference
on this host — `"f32"` or `"bf16"`. Implemented in the shim: `dlopen` the vendored OpenVINO C
library, `ov_core_create`, `ov_core_get_property("CPU", "INFERENCE_PRECISION_HINT")`, free.

The Python consumer reads this through `openvino.Core()` from the installed wheel. We have no such
path — the vendored C API is the only OpenVINO we have — which is why this goes through JNI.

**The caveat, carried over intact:** this reports what a **freshly created** Core would choose on
this host, not a reading from the Core the delegate built inside `OpenvinoBackend`. Those agree
today because the choice derives from CPU capability alone. If per-model precision control is ever
added they could diverge, and this would have to read through the delegate instead — which
ExecuTorch exposes no way to do today.

Two constraints specific to us. Creating a Core loads the CPU plugin and is not cheap, so this is an
on-demand diagnostic that must never touch the hot path or model load. And when the bundle is
absent — no jar, or a platform with no delegate — it returns the literal string `"unavailable"`
rather than throwing, so a monitoring caller degrades the same way the stats surface does with its
`-1`. A diagnostic that throws is a diagnostic people stop calling.

Its real job is keeping the parity tolerance honest: `atol=1e-2` alone cannot distinguish "correct
in bf16" from "quietly degraded", so we record which one we saw.

## Fixture

Vendored from the upstream release asset `etnp-openvino-fixtures-1.3.1-2025.4.1.tar.gz`:
`openvino_tiny.pte`, `in.bin`, `out.bin`, `shape`. The model is `Linear(8,8) + relu`, fully
delegated, `CompileSpec("device", b"CPU")`. `in.bin` and `out.bin` are float32, 32 bytes for 8
values; the tensor shape is read from `method_meta` rather than hardcoded.

The four members are unpacked into `src/test/resources/models/openvino/` and **committed**, matching
this repo's fixture convention and keeping tests offline. Beside them sits a `MANIFEST` recording the
upstream tarball URL, its SHA256, and both coupled versions, so the coupling survives even though
the bytes are vendored.

**`out.bin` is the eager golden, not the delegate's output**, so parity is a tolerance check.

## Testing

**Every OpenVINO test runs in its own JVM.** `OPENVINO_LIB_PATH` is process environment and the
delegate's `dlopen` is once-only, so cases sharing a JVM contaminate each other in ways that present
as flakes. `stressSweepCore` / `stressSweepBaseline` is the existing precedent for a forked test task
in this build.

| Test | Asserts |
|---|---|
| fixture parity | outputs match `out.bin` at `atol=1e-2`; the chosen precision is printed |
| off-platform error | the typed error naming the platform — runs where the backend is *absent* |
| bundle-absent error | the typed error naming the missing artifact |
| `OPENVINO_LIB_PATH` validation | pointing it at a directory raises our error, not the delegate's |
| caller override | an already-valid value is honoured untouched |
| concurrent extraction | two JVMs racing converge on one published directory |

The aarch64 leg runs the **inverse** test — that a `.pte` using an unlinked delegate produces our
platform error — so both matrix legs assert something real instead of one merely skipping.

### On the tolerance

**`atol=1e-2`, and it must not be tightened.** OpenVINO selects inference precision from the CPU it
lands on, at import time rather than at blob-compile time. On bf16-capable hardware (avx512_bf16 /
AMX) the delegate computes in bf16 and lands ~2.5e-3 from the f32 eager golden; elsewhere it lands at
~6e-8. Both are correct OpenVINO results. A tolerance drawn between them asserts which machine CI
happened to allocate — a property this project does not own — and fails at random. Upstream hit
exactly this (`ea393da` in `executorch-runtime-dist`) after a green run and a red run on identical
artifacts.

The tolerance carries this reason as a comment, or a future reader will "fix" it. The loose bound
still catches everything it exists to catch: a delegate returning zeros, garbage, or the wrong model
is orders of magnitude out.

## Version coupling

The OpenVINO version comes to live in three places: `ET_RUNTIME_OPENVINO_VERSION` in the pin, the
bundle `MANIFEST` in the jar, and the fixture `MANIFEST`. They must agree, or a rebuild can vendor a
runtime that cannot import the fixture's precompiled blob — surfacing at model load rather than at
build time.

A shell check in `native/tests/` asserts the three agree. That directory's convention is behavioural
checks and policy bans rather than greps on current wording, and this is a behavioural check: it
fails when the versions genuinely diverge.

OpenVINO versions independently of ExecuTorch, so an OV re-roll can invalidate the committed fixture
with no ET bump. This check is what makes that a build failure instead of a runtime mystery.

## Portability: what a Windows delegate would require

The OpenVINO *runtime bundle* is linux-x86_64-only today (the delegate is not — see the correction
at the top), and upstream is actively exploring Windows. Nothing in this
design may hardcode that assumption, and the parts that would still have to change are recorded here
so the port is a known quantity rather than a discovery.

**Already portable by construction:**

- CMake keys on `if(TARGET openvino_backend)`, never on `ET_PLATFORM`.
- The pin publishes `ET_RUNTIME_OPENVINO_PLATFORM`; the fetch guard compares it against the row being
  built rather than testing for a platform name.
- Java resolves `/native/<platform>/openvino/` off the same `LibUtils.platform()` it already uses, so
  a new bundle row needs no Java change.
- `LibUtils.cacheRoot()` already handles `LOCALAPPDATA`.
- Atomic directory publish plus "never load out of the staging directory" holds on a platform that
  refuses to delete a loaded library.

**Known deltas, none of them solved here:**

- **`$ORIGIN` is a Linux mechanism.** The flat-directory contract works because every library carries
  `RPATH=$ORIGIN`. Windows resolves dependencies by DLL search order instead, and whether a
  `LoadLibrary` of the C API finds its siblings is upstream's problem to solve in how the bundle is
  built, not something this engine can paper over.
- **`setenv` does not exist on Windows.** The equivalent is `_putenv_s`, which writes the **CRT**
  environment, while `GetEnvironmentVariable` reads the **Win32** one. They are different blocks. We
  happen to be safe because our shim and the statically linked runtime are a single DLL sharing one
  CRT copy under `/MT` — but that is a load-bearing accident, and splitting them would break it
  silently.
- **No symlink.** See below; the design should never need one on any platform.
- The `-static` (`/MT`) row selection already documented for the runtime tarball would apply to any
  OpenVINO bundle too.

## Verify during implementation, do not assume

1. ~~Whether the symlink is needed at all.~~ **Settled: it is not.** Measured on the shipped
   `openvino-runtime-2025.4.1-linux-x86_64` bundle — with `libopenvino_c.so` deleted from a flat
   extraction directory, `dlopen("<dir>/libopenvino_c.so.2541")` succeeds, the whole dependency
   graph resolves through `$ORIGIN`, and `ov_core_create` + `ov_core_get_property` return `Ok`. So
   the design points `OPENVINO_LIB_PATH` at the versioned file and never creates a symlink, which
   removes an extraction bug class and one Windows blocker. The ABI suffix is not hardcoded: the
   bundle ships a `BUILDINFO` carrying `ov_abi=2541`.
2. **The size cost of linking `libopenvino_backend.a` into the standard shim.** Measure the `.so`
   growth against the pre-change build. If it is small, one shim with capability-driven linking is
   far simpler than two shim variants; if it is not, the variant question reopens.
3. **`out.bin` dtype and shape**, from the emitter's contract rather than the unpacked tarball.
4. **`MethodMeta::uses_backend` does not load the method.** Measured in the sibling project at ~8 µs.
   See the note under References — the guard that project uses to keep this honest is weaker than it
   appears, so ours should assert the timing directly rather than copy theirs.
5. **Concurrent extraction** actually converges, under a real two-JVM race rather than a unit test of
   the rename.

## Out of scope, permanently

Everything AOT: export, `OpenvinoPartitioner`, `OpenVINOQuantizer`. Also out: non-CPU OpenVINO
devices — the fixture is `CompileSpec("device", b"CPU")` and CPU is the only plugin upstream ships —
and any platform where upstream ships no delegate. That last is a capability statement, not a
platform list, and is expected to change.

## References

- `docs/openvino-jni-consumer.md` (upstream) — the `LD_LIBRARY_PATH` impossibility, what to vendor,
  the `libopenvino_ir_frontend` trap, the glibc floor, version compatibility
- `docs/xnnpack-workspace-size-consumer.md` (upstream) — the backend-option contract
- `ea393da` (upstream) — why the tolerance is 1e-2 and why the precision is reported
- `executorch-numpy-runtime`, `docs/superpowers/specs/2026-08-16-openvino-linux-x86_64-design.md` —
  the sibling design this borrows its detection strategy, error taxonomy, and tolerance reasoning
  from. Its packaging section does not transfer: a Python consumer resolves the runtime from an
  installed wheel and explicitly rules vendoring out of scope, which is the one problem we cannot
  avoid.
- `runtime/executor/method_meta.h:252-273` — `uses_backend` / `num_backends` / `get_backend_name`

**A correction to carry when reading the sibling's material.** Both its spec and the upstream
consumer doc state that the XNNPACK workspace arena is created lazily "during delegate init". That is
not true on this engine's path, measured during the workspace-metric work: `EtRuntime`'s constructor
calls `load_forward()` unconditionally, so delegate init has already run when it returns, and the
arena is still 0. It grows on the first **execute**. The sibling's tests pass anyway because they
load *and run* their fixture — but its `method_meta` timing guard, which asserts the workspace is 0
after a `method_meta` call to prove detection precedes delegate init, is weaker than intended: that
assertion would also hold if `method_meta` did eagerly load the method, since loading alone does not
grow the arena either. Do not copy that guard as-is.
