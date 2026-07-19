cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ET_BUILD_QA "Build native QA targets (Catch2 units + leak harness)" OFF)
option(ET_BUILD_BENCH "Build the Release timing harness (no sanitizers)" OFF)

# --- Runtime resolution: escape hatch (ET_INSTALL set) OR FetchContent the pinned tarball. ---
# The ExecuTorch runtime is NOT built here; it is downloaded (hash-pinned + attested by Repo A).
# See docs/executorch-build-notes.md. ET_PRINT_RESOLUTION is a host-test seam: it prints the
# resolved config and returns before any download or find_package.
option(ET_PRINT_RESOLUTION "Print resolved runtime config and exit (host-test seam)" OFF)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EtRuntimePin.cmake)
set(ET_RUNTIME_VARIANT "logging" CACHE STRING "Runtime variant: logging (ship) | bare | devtools (bench)")
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
else()
  set(_ET_PLATFORM_DEFAULT "linux-x86_64")
endif()
set(ET_PLATFORM "${_ET_PLATFORM_DEFAULT}" CACHE STRING "Platform identity: windows-x86_64 | linux-x86_64")

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
  # windows-x86_64 ships `logging` only; bare/devtools are Linux-only benchmarking builds that upstream
  # never produced for Windows. Without this, the generic "No pin row" error below reads like a broken
  # pin rather than a platform that never had those artifacts. Inside the fetch branch on purpose: the
  # ET_INSTALL escape hatch consults no pin row, so the variant is irrelevant there.
  if(ET_PLATFORM STREQUAL "windows-x86_64" AND NOT ET_RUNTIME_VARIANT STREQUAL "logging")
    message(FATAL_ERROR
      "windows-x86_64 ships the 'logging' variant only; got '${ET_RUNTIME_VARIANT}'. "
      "bare/devtools are Linux-only (benchmarking; see native/build_variants.sh).")
  endif()
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${ET_RUNTIME_ROW}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${ET_RUNTIME_ROW}}")
  if(NOT _ET_RESOLVED_URL)
    message(FATAL_ERROR "No pin row for variant='${ET_RUNTIME_VARIANT}' row='${ET_RUNTIME_ROW}' (platform='${ET_PLATFORM}') in EtRuntimePin.cmake")
  endif()
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

# --- JNIEnv-free core: shared by the .so, the units, and the leak harness ---
add_library(et_runtime STATIC ${CMAKE_CURRENT_SOURCE_DIR}/core/et_runtime.cpp)
target_include_directories(et_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
set_target_properties(et_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Bare target names; backend/ops self-whole-archive — link plainly. PUBLIC so consumers inherit.
target_link_libraries(et_runtime PUBLIC
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)

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
  endif()
endif()

# --- Release timing harness (no sanitizers). OFF for the shipping build and for QA. ---
if(ET_BUILD_BENCH)
  add_executable(et_timing_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_timing_harness.cpp)
  target_link_libraries(et_timing_harness PRIVATE et_runtime)
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
  target_compile_definitions(et_runtime_test PRIVATE
    ADD_PTE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/spike/add.pte")
  target_include_directories(et_runtime_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jni)

  add_executable(et_leak_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_leak_harness.cpp)
  target_link_libraries(et_leak_harness PRIVATE et_runtime)
endif()
