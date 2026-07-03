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
set(_ET_PLATFORM "linux-x86_64")

if(ET_INSTALL)
  set(_ET_RESOLUTION "escape-hatch")
  set(_ET_STEM "")
  set(_ET_RESOLVED_URL "")
else()
  set(_ET_RESOLUTION "fetch")
  set(_ET_STEM "executorch-runtime-${ET_RUNTIME_ET_VERSION}-${ET_RUNTIME_VARIANT}-${_ET_PLATFORM}")
  set(_ET_RESOLVED_URL "${ET_RUNTIME_URL_${ET_RUNTIME_VARIANT}_${_ET_PLATFORM}}")
  set(_ET_RESOLVED_SHA "${ET_RUNTIME_SHA256_${ET_RUNTIME_VARIANT}_${_ET_PLATFORM}}")
  if(NOT _ET_RESOLVED_URL)
    message(FATAL_ERROR "No pin row for variant='${ET_RUNTIME_VARIANT}' platform='${_ET_PLATFORM}' in EtRuntimePin.cmake")
  endif()
endif()

if(ET_PRINT_RESOLUTION)
  message(STATUS "ET_RESOLUTION resolution=${_ET_RESOLUTION} variant=${ET_RUNTIME_VARIANT} stem=${_ET_STEM} url=${_ET_RESOLVED_URL} et_install=${ET_INSTALL}")
  return()
endif()

if(_ET_RESOLUTION STREQUAL "fetch")
  include(FetchContent)
  FetchContent_Declare(et_runtime URL "${_ET_RESOLVED_URL}" URL_HASH "SHA256=${_ET_RESOLVED_SHA}")
  FetchContent_MakeAvailable(et_runtime)
  # The tarball has a single top-level dir (the stem) under the extracted source dir.
  set(ET_INSTALL "${et_runtime_SOURCE_DIR}/${_ET_STEM}")
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
  target_include_directories(executorch_djl PRIVATE
    "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
  target_link_libraries(executorch_djl PRIVATE et_runtime)
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
