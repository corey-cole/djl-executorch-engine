package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.measly.executorch.TestSupport;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class EtNativeTest {
    private static EtTensor floatScalar(float v) {
        ByteBuffer b = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
        b.putFloat(0, v);
        return new EtTensor(new long[] {1}, 6 /*Float*/, b);
    }

    @Test
    void forwardAddsTwoScalars() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            FlatInputs in = FlatInputs.of(floatScalar(2f), floatScalar(3f));
            EtTensor[] out = EtNative.forward(
                    handle, in.flatShapes, in.shapeOffsets, in.scalarTypes, in.buffers);
            assertEquals(1, out.length);
            assertArrayEquals(new long[] {1}, out[0].shape);
            assertEquals(6, out[0].scalarType);
            assertEquals(5f, out[0].data.order(ByteOrder.nativeOrder()).getFloat(0), 1e-6);
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void twoArgLoadModuleLoadsWithProfilingOff() {
        TestSupport.assumeNativeAvailable();
        // The two-argument form exists so that adding the profiling parameter removed no signature.
        // Its contract is that it loads a working model with the tracer OFF, so this asserts both:
        // the forward still computes, and the model produced no dump.
        assumeTrue(
                EtNative.devtoolsAvailable(),
                "an unprofiled model dumps nothing either way where devtools is absent;"
                        + " the empty dump only discriminates where the tracer could have recorded");
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1);
        try {
            FlatInputs in = FlatInputs.of(floatScalar(2f), floatScalar(3f));
            EtTensor[] out = EtNative.forward(
                    handle, in.flatShapes, in.shapeOffsets, in.scalarTypes, in.buffers);
            assertEquals(5f, out[0].data.order(ByteOrder.nativeOrder()).getFloat(0), 1e-6);
            // Had the overload passed profiling=true, this forward would have recorded an
            // Execute block and the dump would be non-empty.
            assertEquals(0, EtNative.etDump(handle).length, "the overload must not attach a tracer");
        } finally {
            EtNative.destroy(handle);
        }
    }
}
