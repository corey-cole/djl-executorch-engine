# Post-link guard: prove the XNNPACK backend registration survived the final .so link.
#
# The XNNPACK backend registers itself from a pure static-initializer TU
# (XNNPACKBackend.cpp) inside libxnnpack_backend.a. Nothing references its symbols, so a
# normal archive link — especially with --gc-sections — GC's the TU away and you get a
# silent "backend not found" only at model-load time, long after everything compiled and
# linked cleanly. The runtime's ExecuTorchTargets.cmake self-whole-archives xnnpack_backend
# to prevent this (see native/CMakeLists.txt link site), but that guarantee lives in the
# *runtime tarball* we download and can regress out from under us (e.g. a re-rolled Repo A
# tarball whose config drops the INTERFACE_LINK_OPTIONS wrap). This turns that regression
# into a build failure instead of a runtime one. See docs/handover-to-engine-2.md §6.
#
# Invoked via: cmake -DSO=<lib> -DNM=<nm> -P assert_xnnpack_registered.cmake

if(NOT SO OR NOT EXISTS "${SO}")
  message(FATAL_ERROR "assert_xnnpack_registered: SO not found: '${SO}'")
endif()
if(NOT NM)
  set(NM "nm")
endif()

execute_process(
  COMMAND "${NM}" "${SO}"
  OUTPUT_VARIABLE _syms
  RESULT_VARIABLE _rc
  ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "assert_xnnpack_registered: '${NM} ${SO}' failed (rc=${_rc}): ${_err}")
endif()

# 1) The backend-registration static-init TU must be present. If whole-archive failed and
#    --gc-sections dropped it, this symbol vanishes.
string(FIND "${_syms}" "_GLOBAL__sub_I_XNNPACKBackend" _reg_pos)
if(_reg_pos EQUAL -1)
  message(FATAL_ERROR
    "XNNPACK backend registration was dropped from ${SO}: "
    "static-initializer '_GLOBAL__sub_I_XNNPACKBackend.cpp' is absent.\n"
    "The xnnpack_backend archive was not whole-archive'd at the final link, so the "
    "backend will fail to register at model-load time ('backend not found').\n"
    "Check that the downloaded runtime's ExecuTorchTargets.cmake still self-whole-archives "
    "xnnpack_backend (see docs/handover-to-engine-2.md §6).")
endif()

# 2) Defense in depth: the XNNPACK microkernels themselves must be present, not just the
#    registration shell. A whole-archive'd registration TU with GC'd microkernels would still
#    fail at execute time.
string(REGEX MATCHALL "[^\n]* [Tt] [^\n]*xnn_[^\n]*" _xnn_text "${_syms}")
list(LENGTH _xnn_text _xnn_count)
if(_xnn_count LESS 100)
  message(FATAL_ERROR
    "XNNPACK microkernels look dropped from ${SO}: only ${_xnn_count} defined 'xnn_*' text "
    "symbols (expected hundreds). The XNNPACK archive was not linked whole. "
    "See docs/handover-to-engine-2.md §6.")
endif()

message(STATUS "XNNPACK post-link assertion OK: backend registration present, ${_xnn_count} xnn_* text symbols")
