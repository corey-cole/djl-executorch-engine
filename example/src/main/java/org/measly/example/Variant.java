package org.measly.example;

import java.util.List;
import java.util.function.Function;

/**
 * The three benchmark/example arms and how to build each. Shared by {@link MobilenetExample} and
 * the JMH benchmark so both agree on engine + translator per arm.
 *
 * <p>Factories are method references, so a {@code MobilenetTranslator} (which allocates a PyTorch
 * {@code NDManager} on construction) is built only when {@link #newTranslator(List)} is actually
 * called for {@code ET_HYBRID}/{@code PYTORCH} — never at enum init, and never for {@code
 * ET_NATIVE}. That keeps an {@code ET_NATIVE}-only path (its own JMH fork) free of LibTorch.
 */
enum Variant {
    ET_HYBRID("ExecuTorch", MobilenetTranslator::new),
    PYTORCH("PyTorch", MobilenetTranslator::new),
    ET_NATIVE("ExecuTorch", PlainJavaMobilenetTranslator::new);

    final String engine;
    private final Function<List<String>, ? extends CloseableImageTranslator> factory;

    Variant(String engine, Function<List<String>, ? extends CloseableImageTranslator> factory) {
        this.engine = engine;
        this.factory = factory;
    }

    /** Builds this arm's translator. Caller owns closing it. */
    CloseableImageTranslator newTranslator(List<String> synset) {
        return factory.apply(synset);
    }
}
