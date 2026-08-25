/**
 * The JNI boundary to the ExecuTorch native library — <b>internal, not supported API</b>.
 *
 * <p>Every type here is {@code public} because JNI linkage and cross-package access inside the
 * engine require it, not because it is offered to callers. Signatures in this package may change
 * in any release, including a patch release, with no deprecation cycle: they track the native
 * shim, whose shape is decided by the runtime rather than by compatibility.
 *
 * <p>The supported entry points are {@link org.measly.executorch.engine.EtEngine} and {@link
 * org.measly.executorch.engine.EtModel}, reached through DJL's {@code Criteria}. Anything
 * obtainable here is obtainable there.
 */
package org.measly.executorch.jni;
