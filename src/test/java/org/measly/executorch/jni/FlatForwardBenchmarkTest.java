package org.measly.executorch.jni;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.measly.executorch.TestSupport;

/**
 * SPIKE (branch spike/flat-array-forward): A/B benchmark of {@link EtNative#forward(long,
 * EtTensor[])} against {@link EtNative#forwardFlat}. Not meant to be committed to main.
 */
class FlatForwardBenchmarkTest {

    private static final int N = 41;

    @Test
    void compareForwardVariants() {
        TestSupport.assumeNativeAvailable();
        Assumptions.assumeTrue(new File("native/spike/add41.pte").isFile());
        long handle = EtNative.loadModule(
                new File("native/spike/add41.pte").getAbsolutePath(), -1, false);
        try {
            EtTensor[] tensors = new EtTensor[N];
            long[] flatShapes = new long[N];
            int[] shapeOffsets = new int[N + 1];
            int[] scalarTypes = new int[N];
            ByteBuffer[] buffers = new ByteBuffer[N];
            for (int i = 0; i < N; i++) {
                ByteBuffer b = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder());
                b.putFloat(0, 1f);
                tensors[i] = new EtTensor(new long[] {1}, 6, b);
                flatShapes[i] = 1;
                shapeOffsets[i + 1] = i + 1;
                scalarTypes[i] = 6;
                buffers[i] = b;
            }

            int warmup = 20000;
            int iters = 200000;

            for (int i = 0; i < warmup; i++) {
                EtNative.forwardFlat(handle, flatShapes, shapeOffsets, scalarTypes, buffers);
            }
            long t1 = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                EtNative.forwardFlat(handle, flatShapes, shapeOffsets, scalarTypes, buffers);
            }
            double meanFlat = (double) (System.nanoTime() - t1) / iters;

            for (int i = 0; i < warmup; i++) {
                EtNative.forward(handle, tensors);
            }
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) {
                EtNative.forward(handle, tensors);
            }
            double meanArrayOfStruct = (double) (System.nanoTime() - t0) / iters;

            System.out.println("ADHOC_TIMING variant=arrayOfStruct mean_ns_per_forward="
                    + meanArrayOfStruct + " iters=" + iters);
            System.out.println("ADHOC_TIMING variant=flat mean_ns_per_forward="
                    + meanFlat + " iters=" + iters);
        } finally {
            EtNative.destroy(handle);
        }
    }
}
