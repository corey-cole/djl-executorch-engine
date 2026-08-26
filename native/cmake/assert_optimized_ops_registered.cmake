# Post-link guard: prove optimized_native_cpu_ops_lib's kernels survived the final .so link.
#
# Unlike xnnpack_backend (assert_xnnpack_registered.cmake), this archive's static-initializer TU
# is NOT a usable signal: it is named RegisterCodegenUnboxedKernelsEverything.cpp, the exact same
# generated-codegen filename portable_ops_lib's own registration TU uses, so the two produce an
# identical mangled `_GLOBAL__sub_I_...` symbol. That symbol's mere presence proves nothing here —
# note this library replaces portable_ops_lib in the link rather than joining it (see the
# native/CMakeLists.txt link site: the two register overlapping (name_, kernel_key_) pairs and
# linking both aborts the process at .so-load time), so there's no "other" copy of that symbol to
# confuse this check the way there would be if both were linked, but the pattern is still not a
# distinguishing one to build a guard on.
#
# The archive IS distinguishable by kernel content: its Eigen-backed specializations are named
# with an `opt_` prefix (opt_add_out, opt_mul_out, opt_linear_out, ...) that portable_ops_lib never
# uses (verified zero matches against the pinned runtime's libportable_ops_lib.a). Counting those
# symbols in the final .so is this guard's signal, mirroring the xnn_* microkernel count check's
# role in the XNNPACK guard.
#
# Invoked via: cmake -DSO=<lib> -DNM=<nm> -P assert_optimized_ops_registered.cmake

if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_optimized_ops_registered: SO not found: '${SO}'")
endif()
if(NOT NM)
  set(NM "nm")
endif()

execute_process(
  COMMAND "${NM}" "-C" "${SO}"
  OUTPUT_VARIABLE _syms
  RESULT_VARIABLE _rc
  ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "assert_optimized_ops_registered: '${NM} -C ${SO}' failed (rc=${_rc}): ${_err}")
endif()

string(REGEX MATCHALL "[^\n]* [Tt] [^\n]*::opt_[^\n]*_out\\(" _opt_text "${_syms}")
list(LENGTH _opt_text _opt_count)
if(_opt_count LESS 10)
  message(FATAL_ERROR
    "optimized_native_cpu_ops_lib's kernels look absent from ${SO}: only ${_opt_count} defined "
    "'opt_*_out' text symbols (expected dozens). Either the target was dropped from "
    "target_link_libraries(et_runtime ...) in native/CMakeLists.txt, or the downloaded runtime's "
    "ExecuTorchTargets.cmake stopped self-whole-archiving optimized_native_cpu_ops_lib.")
endif()

message(STATUS "optimized ops post-link assertion OK: ${_opt_count} opt_*_out text symbols")
