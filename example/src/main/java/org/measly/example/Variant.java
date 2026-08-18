package org.measly.example;

import java.util.List;
import java.util.function.Function;

/**
 * The benchmark/example arms and how to build each: engine, model artifact, and translator. Shared
 * by {@link MobilenetExample} and the JMH benchmark so both agree on all three per arm.
 *
 * <p>Factories are method references, so a {@code MobilenetTranslator} (which allocates a PyTorch
 * {@code NDManager} on construction) is built only when {@link #newTranslator(List)} is actually
 * called for the arms that use it — never at enum init, and never for {@code ET_NATIVE}. That keeps
 * an {@code ET_NATIVE}-only path (its own JMH fork) free of LibTorch.
 */
public enum Variant {
    /** ExecuTorch engine (XNNPACK-lowered) with the PyTorch-backed image translator. */
    ET_HYBRID("ExecuTorch", "mobilenet_v2", ".pte", true, MobilenetTranslator::new),

    /** LibTorch/PyTorch engine baseline, same translator. */
    PYTORCH("PyTorch", "mobilenet_v2", ".pt", false, MobilenetTranslator::new),

    /** ExecuTorch engine with a pure-Java translator, so the arm never loads LibTorch. */
    ET_NATIVE("ExecuTorch", "mobilenet_v2", ".pte", true, PlainJavaMobilenetTranslator::new),

    /**
     * ExecuTorch engine with the model lowered to the OpenVINO delegate instead of XNNPACK.
     *
     * <p>Deliberately paired with the same translator as {@link #ET_HYBRID}: holding preprocessing
     * identical is what makes the ET_HYBRID/ET_OPENVINO difference attributable to the delegate.
     *
     * <p>linux-x86_64 only — that is the only platform with a published OpenVINO runtime bundle.
     * Elsewhere the load fails naming the missing runtime.
     */
    ET_OPENVINO("ExecuTorch", "mobilenet_v2_openvino", ".pte", false, MobilenetTranslator::new);

    final String engine;

    private final String baseName;
    private final String extension;

    /**
     * Whether this arm has an {@code alloc_graph_input=False} counterpart export.
     *
     * <p>False for PYTORCH (TorchScript has no such notion) and for ET_OPENVINO (one export, since
     * a fully-delegated graph hands its input straight to the delegate). Arms that say false use
     * the same artifact for every {@code exportMode}, so those benchmark cells duplicate rather
     * than fail on a missing file.
     */
    private final boolean hasUnplannedExport;

    private final Function<List<String>, ? extends CloseableImageTranslator> factory;

    Variant(
            String engine,
            String baseName,
            String extension,
            boolean hasUnplannedExport,
            Function<List<String>, ? extends CloseableImageTranslator> factory) {
        this.engine = engine;
        this.baseName = baseName;
        this.extension = extension;
        this.hasUnplannedExport = hasUnplannedExport;
        this.factory = factory;
    }

    /** @return the DJL model name for this arm's default (memory-planned) export */
    String modelName() {
        return baseName;
    }

    /** @return the artifact file name for this arm's default (memory-planned) export */
    String artifact() {
        return baseName + extension;
    }

    /**
     * @param exportMode {@code "planned"} or {@code "unplanned"}
     * @return the DJL model name for this arm under the given export mode, falling back to the
     *     default export for arms that have no unplanned counterpart
     */
    String modelName(String exportMode) {
        return hasUnplannedExport && "unplanned".equals(exportMode) ? baseName + "_unplanned" : baseName;
    }

    /**
     * @param exportMode {@code "planned"} or {@code "unplanned"}
     * @return the artifact file name for this arm under the given export mode
     */
    String artifact(String exportMode) {
        return modelName(exportMode) + extension;
    }

    /** Builds this arm's translator. Caller owns closing it. */
    CloseableImageTranslator newTranslator(List<String> synset) {
        return factory.apply(synset);
    }
}
