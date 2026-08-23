# Bumping the vendored OpenVINO version

OpenVINO versions independently of ExecuTorch: a runtime-dist release can change
`ET_RUNTIME_OPENVINO_VERSION` without changing the ExecuTorch version, and an ExecuTorch bump can
leave OpenVINO untouched. So this is its own procedure, not a step inside a pin bump.

The failure this prevents is specific. A `.pte` embeds a **precompiled OpenVINO blob**, so vendoring
a runtime the committed fixture's blob cannot be imported by fails at *model load* with
`failed to import model for device 'CPU'` — a message naming none of the causes below.

## What must change together

1. **`native/cmake/EtRuntimePin.cmake`** — generated. Replace it wholesale with the asset from the
   new runtime-dist release; do not hand-edit.
   This carries `ET_RUNTIME_OPENVINO_VERSION` and the per-platform
   `ET_RUNTIME_OPENVINO_URL_<platform>` / `ET_RUNTIME_OPENVINO_SHA256_<platform>` rows.
2. **`src/test/resources/models/openvino/`** — the four fixture members **and** their `MANIFEST`.
   The fixture asset is OpenVINO-version-coupled by name
   (`etnp-openvino-fixtures-<etver>-<ovver>.tar.gz`), so a new OpenVINO means a new fixture. Update
   `openvino_version`, `tarball_url`, and `tarball_sha256` in the `MANIFEST` to match the asset you
   actually unpacked.
3. **`native/tests/openvino_bundle_staging.sh`** — only if the bundle's library set changed. It
   holds the expected member set per platform, and it is the only place that enumeration lives:
   `OpenVinoRuntime` reads whatever the staged `MANIFEST` declares, so nothing in Java needs
   editing. A count mismatch from this test is the signal to come here.

## What must NOT change

- **The ABI suffix is never hardcoded anywhere.** It tracks the version (`2025.4.1` → `2541`) and is
  read from the bundle's `BUILDINFO` (`ov_abi`) by the staging test's Linux arm; Java reads
  `c_library` from the `MANIFEST`. If you find yourself editing a `2541` literal, something has
  regressed.
- **No symlink is ever created.** `OPENVINO_LIB_PATH` names the versioned file; `$ORIGIN` resolves
  the rest. Verified against the shipped bundle.
- **`atol=1e-2` in the parity test.** A new OpenVINO does not justify tightening it — the bound is
  about which CPU the test lands on, not which version it runs.
- **No Java file lists the bundle's libraries.** `MANIFEST`'s `libs` and `c_library` are generated
  from the staged tree by `native/build.sh`. If you find yourself adding a filename to a `List<String>`
  in `OpenVinoRuntime`, something has regressed.

## Verifying the bump

```bash
./native/local_build_wrapper.sh                             # restages the bundle from the new pin
./native/tests/openvino_version_coupling.sh                 # pin == fixture == staged bundle
./native/tests/openvino_bundle_staging.sh                   # library set and ABI derivation
./native/tests/openvino_bundle_staging.sh windows-x86_64    # on winbox, after staging there
./gradlew openvinoTest                                      # parity against the new fixture
```

If parity fails but everything else passes, the fixture and the runtime disagree — you almost
certainly updated one of items 1 and 2 without the other.
