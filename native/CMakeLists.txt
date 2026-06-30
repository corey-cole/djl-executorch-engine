cmake_minimum_required(VERSION 3.24)
project(executorch_djl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ET_BUILD_QA "Build native QA targets (Catch2 units + leak harness)" OFF)

# TODO: ET_INSTALL needs to cover the entire path and should default to something that
# is compatible with local and GHA builds
set(ET_INSTALL "$ENV{HOME}/workspace/executorch/cmake-out" CACHE PATH "ExecuTorch install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${ET_INSTALL}")
set(tokenizers_DIR "${ET_INSTALL}/lib/cmake/tokenizers" CACHE PATH "")
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
target_link_libraries(executorch_djl PRIVATE et_runtime)

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

  add_executable(et_leak_harness ${CMAKE_CURRENT_SOURCE_DIR}/harness/et_leak_harness.cpp)
  target_link_libraries(et_leak_harness PRIVATE et_runtime)
endif()
