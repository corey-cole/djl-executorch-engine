#!/usr/bin/env bash
# Regenerates the clangd compile database in native/build-clangd/. See .clangd for why this exists.
#
# Run it once after cloning, and again after any bump of native/cmake/EtRuntimePin.cmake or a
# change to the compile flags in native/CMakeLists.txt -- the database is otherwise never
# refreshed, and a stale one makes clangd resolve against the OLD runtime headers silently.
#
# Configure only: no compilation happens, the database is written at CMake configure time.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_DIR="${REPO_ROOT}/native/build-clangd"

# TWO configures, not one. native/CMakeLists.txt builds the JNI shim only when ET_BUILD_QA and
# ET_BUILD_BENCH are both OFF (the Catch2 units and harnesses are JVM-free and deliberately need
# no JDK), so no single configure covers both jni/ and test/. We merge the two databases below.
cmake -S "${REPO_ROOT}/native" -B "${DB_DIR}/shim" -G Ninja >/dev/null
cmake -S "${REPO_ROOT}/native" -B "${DB_DIR}/qa" -G Ninja -DET_BUILD_QA=ON >/dev/null

python3 - "${DB_DIR}" <<'PY'
import json, pathlib, sys

db = pathlib.Path(sys.argv[1])
merged = {}
# qa first, shim second: core/et_runtime.cpp is compiled by both, and last write wins. The shim
# tree's flags are the ones the shipped library is actually built with, so they should win.
for tree in ("qa", "shim"):
    for entry in json.loads((db / tree / "compile_commands.json").read_text()):
        merged[entry["file"]] = entry
out = db / "compile_commands.json"
out.write_text(json.dumps(list(merged.values()), indent=2))
own = sorted(e["file"].split("/native/")[-1] for e in merged.values() if "_deps" not in e["file"])
print(f"{len(merged)} entries -> {out}")
print("project sources indexed: " + ", ".join(own))
PY
