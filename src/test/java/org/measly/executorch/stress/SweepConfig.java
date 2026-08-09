package org.measly.executorch.stress;

import java.util.ArrayList;
import java.util.List;

/**
 * The sweep matrix, split across the two JVMs that have to run it.
 *
 * <p>The intra-op threadpool is process-global and write-once — sealed at the first model load — so
 * the eight {@code intraOp=1} cells and the {@code intraOp=default} confirmation cell cannot share
 * a process. {@code stressSweepCore} runs {@link #coreCells()} under
 * {@code -Dai.djl.executorch.num_threads=1}; {@code stressSweepBaseline} runs
 * {@link #baselineCells()} with no such flag.
 */
public final class SweepConfig {

    private static final int[] THREAD_COUNTS = {1, 2, 4, 8};

    /** Modes that are not degenerate for a single-model workload. See SweepConfigTest. */
    private static final String[] MODES = {"global", "disabled"};

    private SweepConfig() {}

    /** One measurement cell. {@code intraOp == 0} means "leave the pool at its default size". */
    public record Cell(int threads, String mode, int intraOp) {
        public String label() {
            return "t=" + threads + " mode=" + mode + " intraop=" + (intraOp == 0 ? "default" : intraOp);
        }
    }

    /** Eight cells at one intra-op thread, ordered so the report reads as a scaling curve. */
    public static List<Cell> coreCells() {
        List<Cell> cells = new ArrayList<>(THREAD_COUNTS.length * MODES.length);
        for (int threads : THREAD_COUNTS) {
            for (String mode : MODES) {
                cells.add(new Cell(threads, mode, 1));
            }
        }
        return cells;
    }

    /**
     * The single confirmation cell at the real-world intra-op default, so the sweep cannot be
     * misread as if {@code intraop=1} were the shipping configuration.
     */
    public static List<Cell> baselineCells() {
        return List.of(new Cell(1, "global", 0));
    }
}
