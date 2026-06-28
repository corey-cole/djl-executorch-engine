cmake_minimum_required(VERSION 3.24)
project(et_desktop_spike LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point at the runtime install tree from the Prerequisite ($ET_INSTALL).
set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
# Put the whole install prefix on the search path: executorch-config.cmake internally
# calls find_package(tokenizers CONFIG) with no hint, so tokenizers (a SEPARATE package
# at lib/cmake/tokenizers) must be discoverable or its imported targets go missing.
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

add_executable(cpp_smoke cpp_smoke.cpp)
# Exported targets are bare (no executorch:: namespace in this install) and each
# backend/ops imported target ALREADY carries its own --whole-archive in
# INTERFACE_LINK_OPTIONS. Link them plainly — wrapping again double-includes the
# archive and causes "multiple definition" link errors.
target_link_libraries(cpp_smoke PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)

# --- Task 2: the thin JNI shim shared library ---
# JNI headers ONLY. The .so is loaded BY the JVM, so it must not link libjvm/libjawt;
# find_package(JNI) insists on those libs and fails (missing AWT/JVM). Derive jni.h from
# JAVA_HOME, falling back to the JDK that owns `javac` on PATH.
if(DEFINED ENV{JAVA_HOME})
  set(JAVA_HOME "$ENV{JAVA_HOME}")
else()
  find_program(JAVAC_EXECUTABLE javac REQUIRED)
  get_filename_component(JAVA_HOME "${JAVAC_EXECUTABLE}" DIRECTORY)   # .../bin
  get_filename_component(JAVA_HOME "${JAVA_HOME}" DIRECTORY)          # JDK root
endif()
if(NOT EXISTS "${JAVA_HOME}/include/jni.h")
  message(FATAL_ERROR "jni.h not found under ${JAVA_HOME}/include — set JAVA_HOME to a JDK")
endif()

add_library(executorch_djl SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/../jni/executorch_djl_jni.cpp)
target_include_directories(executorch_djl PRIVATE
  "${JAVA_HOME}/include"
  "${JAVA_HOME}/include/linux")
# Same linking rules as cpp_smoke: bare target names; backend/ops self-whole-archive.
target_link_libraries(executorch_djl PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)