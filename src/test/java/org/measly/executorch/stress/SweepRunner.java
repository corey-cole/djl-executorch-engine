package org.measly.executorch.stress;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Queue;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/** Runs one sweep cell and reports it. Correctness is still checked — a fast wrong answer is not a result. */
public final class SweepRunner {

    private static final Path REPORT = Paths.get("build/reports/stress/sweep.tsv");

    private SweepRunner() {}

    public record Result(
            SweepConfig.Cell cell,
            long forwards,
            double wallSeconds,
            double forwardsPerSecond,
            double meanLatencyMs,
            double achievedParallelism,
            long peakRssKb) {}

    public static Result run(SweepConfig.Cell cell, StressGolden golden, int seconds)
            throws Exception {
        List<StressGolden.Case> cases = golden.cases();

        // Reference pass outside the timed region, verified against the goldens.
        float[][] reference = new float[cases.size()][];
        try (PerThreadContext ctx = PerThreadContext.open(cell.mode())) {
            for (int i = 0; i < cases.size(); i++) {
                reference[i] = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                golden.verify(cases.get(i), reference[i]);
            }
        }

        AtomicBoolean stop = new AtomicBoolean(false);
        AtomicLong forwards = new AtomicLong();
        Queue<Throwable> failures = new ConcurrentLinkedQueue<>();
        CyclicBarrier start = new CyclicBarrier(cell.threads() + 1);
        List<Thread> workers = new ArrayList<>(cell.threads());

        for (int t = 0; t < cell.threads(); t++) {
            Thread worker =
                    new Thread(
                            () -> {
                                try (PerThreadContext ctx = PerThreadContext.open(cell.mode())) {
                                    // Warm up BEFORE the barrier so first-call costs (delegate
                                    // init, page faults, JIT) land outside the timed region.
                                    for (int i = 0; i < cases.size(); i++) {
                                        ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                    }
                                    if (!stop.get()) start.await();
                                    while (!stop.get()) {
                                        for (int i = 0; i < cases.size(); i++) {
                                            float[] out = ctx.predict(cases.get(i).v1, cases.get(i).v2);
                                            if (!Arrays.equals(reference[i], out)) {
                                                throw new AssertionError(
                                                        "cell "
                                                                + cell.label()
                                                                + ": bitwise divergence on case "
                                                                + cases.get(i).name);
                                            }
                                            forwards.incrementAndGet();
                                        }
                                    }
                                } catch (Throwable e) {
                                    failures.add(e);
                                    start.reset(); // release peers parked at the barrier
                                    stop.set(true); // do not let the others spin for the full run
                                }
                            },
                            "sweep-" + cell.threads() + "-" + t);
            workers.add(worker);
            worker.start();
        }

        try {
            start.await(); // release every worker at once; loads and warmup are already done
        } catch (BrokenBarrierException bbe) {
            // A worker failed before reaching the barrier (e.g. open() threw), so the barrier
            // broke instead of tripping. Surface the real failure, not a misleading "Broken barrier".
            if (!failures.isEmpty()) {
                AssertionError e =
                        new AssertionError("cell " + cell.label() + " failed during load/warmup");
                failures.forEach(e::addSuppressed);
                throw e;
            }
            throw bbe;
        }
        long cpu0 = processCpuNanos();
        long t0 = System.nanoTime();
        Thread.sleep(seconds * 1000L);
        stop.set(true);
        for (Thread w : workers) {
            w.join(60_000);
        }
        double wall = (System.nanoTime() - t0) / 1e9;
        double cpuSeconds = (processCpuNanos() - cpu0) / 1e9;

        if (!failures.isEmpty()) {
            AssertionError e = new AssertionError("cell " + cell.label() + " failed");
            failures.forEach(e::addSuppressed);
            throw e;
        }

        long n = forwards.get();
        double throughput = n / wall;
        // Mean per-forward latency as seen by ONE thread: wall time divided by that thread's share.
        double meanLatencyMs = (n == 0) ? 0 : (wall * 1000.0) / (n / (double) cell.threads());
        return new Result(
                cell, n, wall, throughput, meanLatencyMs, cpuSeconds / wall, peakRssKb());
    }

    /** Process CPU time across all threads; differenced to get achieved parallelism. */
    private static long processCpuNanos() {
        java.lang.management.OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        if (os instanceof com.sun.management.OperatingSystemMXBean sun) {
            return sun.getProcessCpuTime();
        }
        return 0;
    }

    /** Peak resident set in KiB from /proc/self/status (VmHWM). Returns 0 off Linux. */
    private static long peakRssKb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmHWM:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", ""));
                }
            }
        } catch (IOException | RuntimeException e) {
            // Not Linux, or an unreadable procfs. RSS is a nice-to-have, not a result.
        }
        return 0;
    }

    /**
     * Prints a table and appends TSV rows. Appends rather than overwrites because the two sweep arms
     * run in separate JVMs (the intra-op pool is process-global and write-once) and both report into
     * this one file.
     */
    public static void report(List<Result> results) {
        StringBuilder tsv = new StringBuilder();
        System.out.println();
        System.out.printf(
                Locale.ROOT,
                "%-32s %12s %12s %10s %10s%n",
                "cell", "fwd/s", "mean ms", "parallel", "peakRSS MB");
        for (Result r : results) {
            System.out.printf(
                    Locale.ROOT,
                    "%-32s %12.1f %12.3f %10.2f %10.1f%n",
                    r.cell().label(),
                    r.forwardsPerSecond(),
                    r.meanLatencyMs(),
                    r.achievedParallelism(),
                    r.peakRssKb() / 1024.0);
            tsv.append(r.cell().threads())
                    .append('\t')
                    .append(r.cell().mode())
                    .append('\t')
                    .append(r.cell().intraOp() == 0 ? "default" : r.cell().intraOp())
                    .append('\t')
                    .append(r.forwards())
                    .append('\t')
                    .append(String.format(Locale.ROOT, "%.3f", r.wallSeconds()))
                    .append('\t')
                    .append(String.format(Locale.ROOT, "%.1f", r.forwardsPerSecond()))
                    .append('\t')
                    .append(String.format(Locale.ROOT, "%.4f", r.meanLatencyMs()))
                    .append('\t')
                    .append(String.format(Locale.ROOT, "%.3f", r.achievedParallelism()))
                    .append('\t')
                    .append(r.peakRssKb())
                    .append('\n');
        }
        try {
            Files.createDirectories(REPORT.getParent());
            boolean fresh = !Files.exists(REPORT);
            if (fresh) {
                Files.writeString(
                        REPORT,
                        "threads\tmode\tintraop\tforwards\twall_s\tfwd_per_s\tmean_ms\tparallelism\tpeak_rss_kb\n",
                        StandardCharsets.UTF_8,
                        StandardOpenOption.CREATE);
            }
            Files.writeString(
                    REPORT, tsv.toString(), StandardCharsets.UTF_8, StandardOpenOption.APPEND);
            System.out.println("appended " + results.size() + " row(s) to " + REPORT);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
