package org.measly.executorch.jni;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.measly.executorch.TestSupport;
import org.junit.jupiter.api.Test;

class EtMethodMetaTest {
    @Test
    void readsAddModelMetadata() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertEquals(2, meta.numInputs);
            assertArrayEquals(new int[] {6, 6}, meta.inputScalarTypes); // two float32 inputs
            assertArrayEquals(new boolean[] {true, true}, meta.inputMemoryPlanned);
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void readsUnplannedAddModelMetadata() {
        TestSupport.assumeUnplannedModelAvailable();
        long handle = EtNative.loadModule(TestSupport.addUnplannedPtePath(), -1, false);
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertEquals(2, meta.numInputs);
            assertArrayEquals(new int[] {6, 6}, meta.inputScalarTypes); // same add model
            assertArrayEquals(new boolean[] {false, false}, meta.inputMemoryPlanned);
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void capturesPlannedArenaBytes() {
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertTrue(
                    meta.plannedArenaBytes > 0,
                    "add.pte is memory-planned, so it must report a planned arena");
        } finally {
            EtNative.destroy(handle);
        }
    }

    @Test
    void capturesInputNamesArray() {
        // Whether a given .pte's tensors carry non-empty debug names is an export-time detail this
        // test does not pin; what must hold regardless is that the array exists, is never null
        // element-for-element, and lines up 1:1 with the other per-input arrays.
        TestSupport.assumeNativeAvailable();
        long handle = EtNative.loadModule(TestSupport.addPtePath(), -1, false);
        try {
            EtMethodMeta meta = EtNative.methodMeta(handle);
            assertEquals(meta.numInputs, meta.inputNames.length);
            for (String name : meta.inputNames) {
                org.junit.jupiter.api.Assertions.assertNotNull(name);
            }
        } finally {
            EtNative.destroy(handle);
        }
    }
}
