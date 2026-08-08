package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import org.measly.executorch.jni.EtNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** ExecuTorch implementation of {@link Engine}. CPU-only, limited NDArray support. */
public final class EtEngine extends Engine {

    public static final String ENGINE_NAME = "ExecuTorch";
    static final int RANK = 10;
    static final String EXECUTORCH_VERSION = "1.3.1"; // pinned ExecuTorch runtime version

    /**
     * JVM flag controlling the intra-op (XNNPACK) thread pool size, e.g.
     * {@code -Dai.djl.executorch.num_threads=2}. Process-global: ExecuTorch's pool is a process
     * singleton (extension::threadpool), so this is NOT a per-model option. Defaults to the
     * performance-core count (cpuinfo-derived), not nproc. There is deliberately no environment
     * variable: verified against the v1.3.1 runtime, nothing reads getenv in extension/threadpool,
     * the vendored pthreadpool, or XNNPACK init, and OMP_NUM_THREADS is inert (no OpenMP symbols).
     * The write window closes at the first model load; see {@link #setIntraOpThreads(int)}.
     */
    public static final String NUM_THREADS_PROPERTY = "ai.djl.executorch.num_threads";

    private static final Logger logger = LoggerFactory.getLogger(EtEngine.class);
    private static final Object INTRAOP_LOCK = new Object();
    // -1 = unset (leave the runtime default); >= 1 = requested. Written by the setter, read at seal.
    private static int pendingIntraOpThreads = -1;
    private static boolean intraOpSealed = false;
    // Effective count decided at seal; -1 = runtime default. Set once, under INTRAOP_LOCK.
    private static int sealedIntraOpThreads = -1;

    private EtEngine() {} // cheap: no native load here (lazy in EtNative)

    static Engine newInstance() {
        return new EtEngine();
    }

    @Override
    public Engine getAlternativeEngine() {
        return null; // Phase 1: no hybrid mode
    }

    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    @Override
    public int getRank() {
        return RANK;
    }

    @Override
    public String getVersion() {
        return EXECUTORCH_VERSION;
    }

    @Override
    public boolean hasCapability(String capability) {
        return false; // no CUDA, no training
    }

    @Override
    public Model newModel(String name, Device device) {
        return new EtModel(name, newBaseManager(device));
    }

    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    @Override
    public NDManager newBaseManager(Device device) {
        return EtNDManager.getSystemManager().newSubManager(device);
    }

    /**
     * Sets the intra-op thread pool size. Process-global and write-once: the value is applied at
     * the first {@code EtModel.load()} (delegate init already submits work to the pool, so that is
     * the only provably safe window for upstream's reset) and later calls throw.
     *
     * @param n pool size; must be >= 1
     * @throws IllegalArgumentException if n &lt; 1 (before any JNI call)
     * @throws IllegalStateException if any model has already been loaded, naming the sealed value
     */
    public static void setIntraOpThreads(int n) {
        if (n < 1) {
            throw new IllegalArgumentException(
                    NUM_THREADS_PROPERTY + " must be >= 1, got " + n);
        }
        synchronized (INTRAOP_LOCK) {
            if (intraOpSealed) {
                throw new IllegalStateException(
                        "Intra-op thread pool is already sealed at "
                                + (sealedIntraOpThreads < 1 ? "the runtime default"
                                                            : String.valueOf(sealedIntraOpThreads))
                                + "; set " + NUM_THREADS_PROPERTY + " before the first model load");
            }
            pendingIntraOpThreads = n;
        }
    }

    /**
     * Effective intra-op pool size as reported by the native pool (get_thread_count), not the
     * requested value -- on a 40-core host the difference is the point. Triggers the native
     * library load.
     */
    public static int getIntraOpThreads() {
        return EtNative.intraOpThreads();
    }

    /**
     * Pure precedence/resolution used by the seal: the setter wins over the property; a present
     * but unparseable or < 1 property is WARNed and ignored (fall back to the runtime default --
     * a typo'd JVM flag must not fail startup). Returns the effective count, or -1 for the
     * runtime default.
     */
    static int resolveIntraOpThreads(int setterValue, String propertyValue) {
        if (setterValue > 0) {
            if (propertyValue != null) {
                logger.warn("{} property is ignored because setIntraOpThreads() was called",
                        NUM_THREADS_PROPERTY);
            }
            return setterValue;
        }
        if (propertyValue == null) {
            return -1;
        }
        try {
            int p = Integer.parseInt(propertyValue.trim());
            if (p >= 1) {
                return p;
            }
            logger.warn("{}={} is < 1; using the runtime default", NUM_THREADS_PROPERTY, propertyValue);
        } catch (NumberFormatException e) {
            logger.warn("{}='{}' is not an integer; using the runtime default",
                    NUM_THREADS_PROPERTY, propertyValue);
        }
        return -1;
    }

    /**
     * Flush point, called by EtModel.load() immediately before the first native call. Under
     * INTRAOP_LOCK: resolves the pending value vs the property (the property is read HERE, at
     * first load -- not at class init -- so the precedence is testable and nothing races
     * EtNative's static initializer), applies it via JNI while still holding the lock (so
     * concurrent loadModel() calls cannot both seal, and no second load's delegate init can race
     * the reset), and marks the pool fixed. Logs the outcome at INFO on this first load.
     */
    static void sealIntraOpThreads() {
        synchronized (INTRAOP_LOCK) {
            if (intraOpSealed) {
                return; // subsequent loads skip the apply; the pool is fixed
            }
            int n = resolveIntraOpThreads(
                    pendingIntraOpThreads, System.getProperty(NUM_THREADS_PROPERTY));
            sealedIntraOpThreads = n;
            intraOpSealed = true;
            if (n >= 1) {
                int actual = EtNative.setIntraOpThreads(n);
                if (actual != n) {
                    logger.warn("{}: requested {} but the pool reports {}; using {}",
                            NUM_THREADS_PROPERTY, n, actual, actual);
                }
                logger.info("intra-op thread pool sealed at {} ({}={})",
                        actual, NUM_THREADS_PROPERTY, n);
            } else {
                logger.info("intra-op thread pool left at the runtime default ({})",
                        EtNative.intraOpThreads());
            }
        }
    }

    /** Sealed effective count (-1 = runtime default). For the unit tests' ISE-message assertion. */
    static int intraOpThreadCount() {
        synchronized (INTRAOP_LOCK) {
            return sealedIntraOpThreads;
        }
    }

    /**
     * Test seam (same package, no-native unit tests only): performs the seal bookkeeping without
     * the JNI apply. Idempotent -- safe whether or not an integration test already sealed this
     * JVM. The native pool is untouched; apply only ever happens in {@link #sealIntraOpThreads()}.
     */
    static void sealIntraOpThreadsForTest() {
        synchronized (INTRAOP_LOCK) {
            if (!intraOpSealed) {
                sealedIntraOpThreads = -1;
                intraOpSealed = true;
            }
        }
    }

    /**
     * Test seam (same package, no-native unit tests only): restores the gate to its unset,
     * unsealed state so each unit test starts from a known position. The native pool is never
     * touched (issue #28).
     */
    static void resetIntraOpForTest() {
        synchronized (INTRAOP_LOCK) {
            pendingIntraOpThreads = -1;
            intraOpSealed = false;
            sealedIntraOpThreads = -1;
        }
    }

    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
