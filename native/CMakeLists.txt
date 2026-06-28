cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
find_package(executorch CONFIG REQUIRED PATHS "${ET_INSTALL}/lib/cmake/ExecuTorch")

# JNI headers only (see Task 2 Step 3 rationale) — never link libjvm/libjawt.
if(DEFINED ENV{JAVA_HOME})
  set(JAVA_HOME "$ENV{JAVA_HOME}")
else()
  find_program(JAVAC_EXECUTABLE javac REQUIRED)
  get_filename_component(JAVA_HOME "${JAVAC_EXECUTABLE}" DIRECTORY)
  get_filename_component(JAVA_HOME "${JAVA_HOME}" DIRECTORY)
endif()

add_library(executorch_djl SHARED ${CMAKE_CURRENT_SOURCE_DIR}/jni/executorch_djl_jni.cpp)
target_include_directories(executorch_djl PRIVATE
  "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
# Bare target names; backend/ops targets self-whole-archive — link plainly (see Task 1 Step 4).
target_link_libraries(executorch_djl PRIVATE
  extension_module_static
  extension_tensor
  xnnpack_backend
  portable_ops_lib
)