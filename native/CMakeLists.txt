cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Emit compile_commands.json from every configure. Load-bearing for native/gen_clangd_db.sh, which
# merges the databases of two configures into the one clangd reads (see .clangd) and passes no
# flags of its own. Ignored by the MSVC/Visual Studio generators, which do not emit the database.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Static CRT (/MT) on Windows: folds the C runtime into the DLL so end users need no VC++
# redistributable. Windows consumers here are developers on potentially locked-down workstations who
# may not be able to run an installer. Pairs with the -static pin row selected below — the runtime
# tarball is itself built /MT (see its BUILDINFO cmake_flags), and every object linked into one
# binary must agree.
#
# A GLOBAL cache variable, not per-target set_property(... MSVC_RUNTIME_LIBRARY ...): Catch2 enters
# the QA build through FetchContent_MakeAvailable, so its targets are not declared here and cannot
# have a property set on them. This variable initialises the property on every target including those
# created in added subdirectories, which is the only mechanism that covers that case. Measured: with
# this set, Catch2.lib carries 106 LIBCMT / 0 MSVCRT directives.
#
# NOT config-dependent (the CMake default is "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"): the pinned
# runtime is Release-only, so a Debug configure must still get the release static CRT rather than
# /MTd, which would not link against it.
#
# CMP0091 must be NEW for this to take effect; cmake_minimum_required(3.24) above guarantees it.
#
# Do NOT expect the linker to police this. Measured upstream: a /MD consumer linked a /MT ExecuTorch
# artifact with no error and not even an LNK4098 warning. The failure is at runtime — two CRTs, two
# heaps, corruption when an allocation crosses the boundary. native/tests/check_windows_crt.sh is the
# real gate.
if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "Static CRT: ship without a VC++ redist")
endif()

option(ET_BUILD_QA "Build native QA targets (Catch2 units + leak harness)" OFF)
option(ET_BUILD_BENCH "Build the Release timing harness (no sanitizers)" OFF)

option(ET_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)

# The check set, as a cache variable so a one-off run can narrow it without editing this file. Two
# additions over GCC's `undefined` umbrella, which excludes both: float-cast-overflow and
# float-divide-by-zero, each reachable in element-count and scale arithmetic. vptr comes off below --
# it needs every TU holding a polymorphic object instrumented, and is the check most likely to
# misfire against the prebuilt runtime, which is not instrumented.
set(ET_UBSAN_CHECKS "undefined,float-cast-overflow,float-divide-by-zero"
    CACHE STRING "UBSan check set passed to -fsanitize=")

# UBSan composes with ASan, so this stacks on the -fsanitize=address flags build_qa.sh passes. It is
# per-TU and local -- each check is an inline test emitted at the operation, with no cross-module
# state -- so linking the uninstrumented prebuilt runtime produces no false positives and hides
# nothing in our own code.
#
# -fno-sanitize-recover=undefined is what makes this a gate rather than a log for the undefined
# umbrella: UBSan's default is print-and-continue, under which CI stays green while diagnostics
# scroll past. The two float checks (float-cast-overflow, float-divide-by-zero) are outside GCC's
# undefined group, so this flag does NOT stop them -- their abort relies on
# UBSAN_OPTIONS=halt_on_error=1 exported by build_qa.sh. Do not drop that export as redundant.
#
# GCC has no ignorelist: -fsanitize-ignorelist and -fsanitize-blacklist are unrecognized, and
# UBSAN_OPTIONS=suppressions= does not suppress these checks. To silence one, use
# __attribute__((no_sanitize("undefined"))) on the function, a per-TU
# set_source_files_properties COMPILE_OPTIONS override, or -fno-sanitize=<check> here -- each with a
# comment naming what was given up and why.
if(ET_UBSAN)
  if(WIN32)
    message(FATAL_ERROR "ET_UBSAN is unsupported on Windows: MSVC has no UndefinedBehaviorSanitizer")
  endif()
  add_compile_options(
      -fsanitize=${ET_UBSAN_CHECKS}
      -fno-sanitize=vptr
      -fno-sanitize-recover=undefined
      -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${ET_UBSAN_CHECKS})
endif()

# --- Runtime resolution: escape hatch (ET_INSTALL set) OR FetchContent the pinned tarball. ---
# The ExecuTorch runtime is NOT built here; it is downloaded (hash-pinned + attested by Repo A).
# See docs/executorch-build-notes.md. ET_PRINT_RESOLUTION is a host-test seam: it prints the
# resolved config and returns before any download or find_package.
option(ET_PRINT_RESOLUTION "Print resolved runtime config and exit (host-test seam)" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EtRuntimePin.cmake)
set(ET_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (fallback) | bare (Linux-only bench) | devtools (shipped on linux-x86_64; platform-keyed via native/variant_select.sh)")
set(ET_INSTALL "" CACHE PATH "Explicit ExecuTorch install prefix (escape hatch); empty => fetch the pinned tarball")
# ET_PLATFORM is the platform IDENTITY; ET_RUNTIME_ROW is the pin-row LOOKUP KEY. They coincide on
# Linux but not on Windows, which publishes two rows for one platform: `windows-x86_64` (/MD, dynamic
# CRT, built for CPython extensions) and `windows-x86_64-static` (/MT). The shim takes the static row
# so the shipped DLL needs no VC++ redistributable.
#
# Keeping identity separate from row is what lets the variant guard below stay a plain STREQUAL. If
# ET_PLATFORM itself carried the `-static` suffix, that guard would silently stop matching and its
# actionable error would degrade into the generic "No pin row" message it exists to pre-empt.
#
# Both are CACHE vars (not plain set) deliberately: it lets the ET_PRINT_RESOLUTION host-test seam
# force a foreign platform (-DET_PLATFORM=windows-x86_64) and assert the pin resolves, with no machine
# of that platform present. See native/tests/cmake_resolution.sh.
if(WIN32)
  set(_ET_PLATFORM_DEFAULT "windows-x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_ET_PLATFORM_DEFAULT "linux-aarch64")
else()
  set(_ET_PLATFORM_DEFAULT "linux-x86_64")
endif()
set(ET_PLATFORM "${_ET_PLATFORM_DEFAULT}" CACHE STRING "Platform identity: windows-x86_64 | linux-x86_64 | linux-aarch64")

# Derive the row default from ET_PLATFORM, NOT from WIN32: the host-test seam sets ET_PLATFORM on a
# Linux box, and deriving from WIN32 there would resolve a Linux row under a Windows identity.
if(ET_PLATFORM STREQUAL "windows-x86_64")
  set(_ET_ROW_DEFAULT "windows-x86_64-static")
else()
  set(_ET_ROW_DEFAULT "${ET_PLATFORM}")
endif()
set(ET_RUNTIME_ROW "${_ET_ROW_DEFAULT}" CACHE STRING "Pin row key in EtRuntimePin.cmake (may differ from ET_PLATFORM)")

if(ET_INSTALL)
  set(_ET_RESOLUTION "escape-hatch")
  set(_ET_STEM "")
  set(_ET_RESOLVED_URL "")
else()
  set(_ET_RESOLUTION "fetch")
  # windows-x86_64 ships `logging` and `devtools` (the pin publishes devtools Windows rows as of
  # v1.4.1-3); `bare` remains Linux-only benchmarking that upstream never produced for Windows.
  # Without this, the generic "No pin row" error below reads like a broken pin rather than a
  # platform that never had those artifacts. Inside the fetch branch on purpose: the ET_INSTALL
  # escape hatch consults no pin row, so the variant is irrelevant there.
  if(ET_PLATFORM STREQUAL "windows-x86_64" AND ET_RUNTIME_VARIANT STREQUAL "bare")
    message(FATAL_ERROR
      "windows-x86_64 ships the 'logging' and 'devtools' variants only; got '${ET_RUNTIME_VARIANT}'. "
      "bare is Linux-only benchmarking (see native/build_variants.sh).")
  endif()
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_RUNTIME_ROW}")
  # The pin file's own selector: it FATAL_ERRORs on a row it does not define, so an unpublished
  # variant/platform pair fails here by name rather than expanding to an empty URL and surfacing
  # later as an opaque FetchContent error.
  et_runtime_dist_url("${ET_RUNTIME_VARIANT}" "${ET_RUNTIME_ROW}" _ET_RESOLVED_URL _ET_RESOLVED_SHA)
endif()

if(ET_PRINT_RESOLUTION)
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} platform=${ET_PLATFORM} row=${ET_RUNTIME_ROW} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
  return()
endif()

if(_ET_RESOLUTION STREQUAL "fetch")
  include(FetchContent)
  FetchContent_Declare(et_runtime URL "${_ET_RESOLVED_URL}" URL_HASH "SHA256=${_ET_RESOLVED_SHA}")
  FetchContent_MakeAvailable(et_runtime)
  # FetchContent strips the tarball's single top-level dir on extraction, so SOURCE_DIR IS the install root.
  set(ET_INSTALL "${et_runtime_SOURCE_DIR}")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
# The tarball ships lib/cmake/{ExecuTorch,tokenizers,absl,re2,pcre2}; prepending CMAKE_PREFIX_PATH
# lets find_package resolve them all. tokenizers_DIR kept for parity with the prior build.
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "" FORCE)
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

# Devtools (ETDump event tracer). Two signals, deliberately, because neither is sufficient alone.
#
# `TARGET etdump` says the runtime tarball shipped the library. It is NOT a promise the tracer was
# compiled in: upstream's add_library(etdump)/install(TARGETS) are unguarded and the root
# CMakeLists adds devtools/ in both branches of if(EXECUTORCH_BUILD_DEVTOOLS), so a tarball could
# carry the target with EXECUTORCH_ENABLE_EVENT_TRACER=OFF. Tracing would then record nothing and
# report success -- the worst failure available.
#
# BUILDINFO's event_tracer key is the distribution's own statement of what it built. It is absent
# under the ET_INSTALL escape hatch, where the target is all we have.
set(ET_EVENT_TRACER_DECLARED "unknown")
if(EXISTS "${ET_INSTALL}/BUILDINFO")
  file(STRINGS "${ET_INSTALL}/BUILDINFO" _et_buildinfo REGEX "^event_tracer=")
  if(_et_buildinfo)
    string(REGEX REPLACE "^event_tracer=" "" ET_EVENT_TRACER_DECLARED "${_et_buildinfo}")
  endif()
endif()

if(TARGET etdump)
  if(ET_EVENT_TRACER_DECLARED STREQUAL "off")
    message(FATAL_ERROR
      "Runtime ships the etdump target but BUILDINFO says event_tracer=off. Linking it would "
      "produce a build that reports profiling support and records nothing. Refusing.")
  endif()
  set(ET_HAVE_DEVTOOLS ON)
  message(STATUS "devtools: event tracer available (BUILDINFO event_tracer=${ET_EVENT_TRACER_DECLARED})")
else()
  if(ET_EVENT_TRACER_DECLARED STREQUAL "on")
    message(FATAL_ERROR
      "BUILDINFO says event_tracer=on but no etdump target was exported. The runtime package is "
      "inconsistent; refusing rather than shipping a build whose capability is unclear.")
  endif()
  set(ET_HAVE_DEVTOOLS OFF)
  message(STATUS "devtools: not provisioned in this runtime (platform=${ET_PLATFORM})")
endif()

# optimized_native_cpu_ops_lib is generated upstream from merged.yaml (kernels/optimized/optimized.yaml
# merged with kernels/portable/functions.yaml as FALLBACK_YAML — see
# executorch/configurations/CMakeLists.txt), so it already registers every op portable_ops_lib does,
# Eigen-backed where a specialization exists (opt_add_out, opt_log_softmax_out, ...) and the plain
# portable kernel otherwise. It REPLACES portable_ops_lib in the link, never joins it: linking both
# duplicates entries under the identical (name_, kernel_key_) key, which
# operator_registry.cpp's register_kernels_internal() treats as a hard registration error —
# ET_CHECK_MSG aborts the process at .so-load static-init time (confirmed: aten::_log_softmax.out
# collided in ./gradlew test).
#
# Not every platform's tarball builds it, though: at this pin, Windows does not (issue #46 tracks
# turning EXECUTORCH_BUILD_KERNELS_OPTIMIZED on there; a first attempt found the /MT build doesn't
# yet compile). `if(TARGET ...)` is the established way this codebase (and upstream itself, e.g.
# executorch/test/openvino/CMakeLists.txt) detects that, rather than branching on platform name.
if(TARGET optimized_native_cpu_ops_lib)
  set(ET_OPS_LIB optimized_native_cpu_ops_lib)
else()
  set(ET_OPS_LIB portable_ops_lib)
endif()

# --- JNIEnv-free core: shared by the .so, the units, and the leak harness ---
add_library(et_runtime STATIC ${CMAKE_CURRENT_SOURCE_DIR}/core/et_runtime.cpp)
target_include_directories(et_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
set_target_properties(et_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Bare target names; backend/ops self-whole-archive — link plainly. PUBLIC so consumers inherit.
target_link_libraries(et_runtime PUBLIC
  extension_module_static
  extension_tensor
  xnnpack_backend
  ${ET_OPS_LIB}
  # Explicit, not transitive through xnnpack_backend: the core now calls
  # get_threadpool()/_unsafe_reset_threadpool itself (spec §5).
  extension_threadpool
  # openVinoInferencePrecision() dlopens/dlsyms the vendored C API (Task 8). The dlopen usage is
  # POSIX-only, and on glibc 2.28 (the manylinux floor) dlsym lives in libdl.so.2, so executables
  # that link et_runtime (Catch2 units, harnesses) need -ldl. The shim tolerates the omission
  # because shared libraries may carry undefined symbols, which is why this only surfaced in QA.
  ${CMAKE_DL_LIBS}
)

# PUBLIC so the Catch2 units and the harnesses inherit both the link and the macro -- they link
# et_runtime and compile code guarded on ET_HAVE_DEVTOOLS.
#
# etdump is linked IN ADDITION TO the ExecuTorch targets above, never instead of them: its exported
# interface is `flatccrt;$<LINK_ONLY:executorch>`, and $<LINK_ONLY:> suppresses usage requirements,
# so C10_USING_CUSTOM_GENERATED_MACROS does not arrive through it. Without that definition
# etdump_flatcc.h reaches a cmake_macros.h that no tarball installs.
if(ET_HAVE_DEVTOOLS)
  target_link_libraries(et_runtime PUBLIC etdump)
  target_compile_definitions(et_runtime PUBLIC ET_HAVE_DEVTOOLS=1)
endif()

# --- JNI shell shared library (loaded BY the JVM: JNI headers only, never libjvm/libjawt) ---
# Built only for the shipping library, NOT for QA builds: the Catch2 units + leak harness are
# JVM-free and link only et_runtime, so a -DET_BUILD_QA=ON configure needs neither the shim nor a
# JDK (no JAVA_HOME / javac required). The shim and QA use separate build dirs, so nothing is lost.
if(NOT ET_BUILD_QA AND NOT ET_BUILD_BENCH)
  if(DEFINED ENV{JAVA_HOME})
    set(JAVA_HOME "$ENV{JAVA_HOME}")
  else()
    find_program(JAVAC_EXECUTABLE javac REQUIRED)
    get_filename_component(JAVA_HOME "${JAVAC_EXECUTABLE}" DIRECTORY)
    get_filename_component(JAVA_HOME "${JAVA_HOME}" DIRECTORY)
  endif()

  add_library(executorch_djl SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/jni/executorch_djl_jni.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/jni/et_logging.cpp)
  # jni_md.h sits in a platform-named subdir: include/linux vs include/win32.
  if(WIN32)
    set(_JNI_MD_DIR "win32")
  else()
    set(_JNI_MD_DIR "linux")
  endif()
  target_include_directories(executorch_djl PRIVATE
    "${JAVA_HOME}/include" "${JAVA_HOME}/include/${_JNI_MD_DIR}")
  target_link_libraries(executorch_djl PRIVATE et_runtime)
  # The OpenVINO delegate ships only in tarballs that were built with it -- linux-x86_64 today.
  # Keyed on the TARGET rather than on ET_PLATFORM so a platform that later gains the delegate works
  # with no edit here; upstream is actively exploring Windows.
  #
  # Deliberately NOT added to the post-link XNNPACK registration assertion: that guard exists to catch
  # the XNNPACK registration being garbage-collected out of the .so, and folding in a target that
  # legitimately does not exist on most platforms would convert a real guard into a platform
  # conditional and weaken it.
  #
  # The archive is static and resolves the OpenVINO C API by dlopen at first use, so this adds no
  # DT_NEEDED and drags in no OpenVINO shared object at link time.
  #
  # Linked PLAINLY, exactly like xnnpack_backend: the runtime's imported openvino_backend target
  # already self-whole-archives via its INTERFACE_LINK_OPTIONS (ExecuTorchTargets.cmake), so a
  # LINK_LIBRARY:WHOLE_ARCHIVE wrap here would link the archive twice and fail with multiple
  # definitions of the OpenvinoBackend members.
  if(TARGET openvino_backend)
    target_link_libraries(executorch_djl PRIVATE openvino_backend)
    message(STATUS "OpenVINO delegate: linked")
  else()
    message(STATUS "OpenVINO delegate: not present in this runtime (platform=${ET_PLATFORM})")
  endif()
  # The shim is dlopen'd by a stock JVM that knows nothing about sanitizer runtimes, so UBSan's
  # runtime has to travel inside the .so. Without this the link succeeds and System.load fails
  # later -- in CI, possibly in a different job on a different machine than the build.
  if(ET_UBSAN)
    target_link_options(executorch_djl PRIVATE -static-libubsan)
  endif()

  # First-party custom ops (etnp LSTM) ship only in some runtime tarballs (Linux logging). The
  # tarball's self-describing ETNPExtras.cmake declares the imported targets + the whole-archive
  # helper. Auto-detect by file presence: the Windows tarball omits it, so the op is skipped there
  # with no platform branch of our own. Whole-archive is required — the op is a static-init
  # registrar; a plain link GCs the registration out (same failure mode the XNNPACK guard defends).
  if(EXISTS "${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
    include("${ET_INSTALL}/lib/cmake/ETNPExtras/ETNPExtras.cmake")
    etnp_extras_whole_archive(executorch_djl)
    message(STATUS "etnp extras: LSTM op whole-archived into executorch_djl")
  else()
    message(STATUS "etnp extras: none in runtime (platform=${ET_PLATFORM}); LSTM op not linked")
  endif()

  # Post-link guard: fail the build if the XNNPACK backend registration got GC'd out of the
  # final .so (the self-whole-archive lives in the downloaded runtime and can regress). Turns a
  # silent load-time "backend not found" into a build failure. See the script header.
  # Linux-only: the guard greps Itanium-ABI static-init mangling out of `nm` (ELF) output. MSVC has
  # neither (.CRT$XCU pointers, no nm). Windows covers the same property at runtime instead — the
  # Catch2 suite executes the XNNPACK-delegated add.pte, which fails if the backend never registered.
  # See docs/superpowers/specs/2026-07-15-windows-builds-design.md §3.4 and §6.
  if(NOT WIN32)
    if(NOT CMAKE_NM)
      find_program(CMAKE_NM NAMES nm)
    endif()
    add_custom_command(TARGET executorch_djl POST_BUILD
      COMMAND ${CMAKE_COMMAND}
        -DSO=$<TARGET_FILE:executorch_djl>
        -DNM=${CMAKE_NM}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_xnnpack_registered.cmake
      VERBATIM
      COMMENT "Asserting XNNPACK backend survived the link")

    # Same regression shape as the XNNPACK guard above, for optimized_native_cpu_ops_lib: catches
    # the target silently falling back out of the link (see native/CMakeLists.txt et_runtime link
    # site and the script header for why a symbol-name check alone can't detect this one). Only
    # meaningful when ET_OPS_LIB actually resolved to optimized_native_cpu_ops_lib -- on a platform
    # where the runtime doesn't ship it (Windows, at this pin) the fallback to plain
    # portable_ops_lib is expected and correct, not a regression, so the opt_*_out symbol count
    # would legitimately be ~0 there.
    if(ET_OPS_LIB STREQUAL "optimized_native_cpu_ops_lib")
      add_custom_command(TARGET executorch_djl POST_BUILD
        COMMAND ${CMAKE_COMMAND}
          -DSO=$<TARGET_FILE:executorch_djl>
          -DNM=${CMAKE_NM}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/assert_optimized_ops_registered.cmake
        VERBATIM
        COMMENT "Asserting optimized ops kernels survived the link")
    endif()
  endif()
endif()

# --- Release timing harness (no sanitizers). OFF for the shipping build and for QA. ---
if(ET_BUILD_BENCH)
  add_executable(et_timing_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_timing_harness.cpp)
  target_link_libraries(et_timing_harness PRIVATE et_runtime)

  # Thread-scaling harness: measures whether concurrent forward() on independent EtRuntime
  # instances actually scales, or serializes on the XNNPACK shared-workspace mutex. Needs an
  # explicit Threads dependency (std::thread/std::barrier) that the single-threaded timing
  # harness does not. See the file header for the knobs and the expected signature.
  find_package(Threads REQUIRED)
  add_executable(et_scaling_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_scaling_harness.cpp)
  target_link_libraries(et_scaling_harness PRIVATE et_runtime Threads::Threads)
endif()

# --- Native QA targets (Catch2 units + ASan/LSan leak harness). OFF for the shipping build. ---
if(ET_BUILD_QA)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.15.1)
  FetchContent_MakeAvailable(Catch2)

  add_executable(et_runtime_test ${CMAKE_CURRENT_SOURCE_DIR}/test/et_runtime_test.cpp)
  target_link_libraries(et_runtime_test PRIVATE et_runtime Catch2::Catch2WithMain)

  # The OpenVINO delegate is NOT pulled in by et_runtime, and the shim -- the only other thing that
  # links it -- is not built under ET_BUILD_QA. Without this the test binary carries no
  # OpenvinoBackend registration, so the ctor guard's FIRST branch ("this build does not provide")
  # fires for every OpenVINO case and the OPENVINO_LIB_PATH branches it is meant to exercise are
  # unreachable. Linked plainly for the same reason the shim does: the imported target
  # self-whole-archives via its INTERFACE_LINK_OPTIONS.
  #
  # ET_OPENVINO_LINKED tells the tests which guard branch to expect. It is genuinely
  # platform-dependent -- linux-aarch64 ships no delegate -- so the tests assert the "not provided"
  # message there and the env-specific messages here, rather than either one skipping.
  if(TARGET openvino_backend)
    target_link_libraries(et_runtime_test PRIVATE openvino_backend)
    target_compile_definitions(et_runtime_test PRIVATE ET_OPENVINO_LINKED=1)
  endif()

  target_compile_definitions(et_runtime_test PRIVATE
    ADD_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/add.pte"
    ADD_UNPLANNED_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/add_unplanned.pte"
    PRIM_INPUT_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/prim_input.pte"
    CONV_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/conv.pte"
    LIN129_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/lin129.pte"
    OPENVINO_TINY_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../src/test/resources/models/openvino/openvino_tiny.pte")
  target_include_directories(et_runtime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jni)

  add_executable(et_leak_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_leak_harness.cpp)
  target_link_libraries(et_leak_harness PRIVATE et_runtime)

  # W4 guard-page harness (manual runs only; never executes in CI — see the brief §3/W4).
  add_executable(et_overread_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_overread_harness.cpp)
  target_link_libraries(et_overread_harness PRIVATE et_runtime)

  # Threaded stress harness: N threads x own EtRuntime over one .pte, asserting bitwise-identical
  # outputs under ASan/LSan. Needs Threads (std::thread/std::barrier), like et_scaling_harness.
  # Always BUILT so it cannot bitrot; only RUN when ET_STRESS=1 (see native/build_qa.sh) so the
  # default QA pass stays cheap.
  find_package(Threads REQUIRED)
  add_executable(et_stress_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_stress_harness.cpp)
  target_link_libraries(et_stress_harness PRIVATE et_runtime Threads::Threads)
endif()
