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

    /** The DJL engine name this plugin registers: {@value}. */
    public static final String ENGINE_NAME = "ExecuTorch";
    static final int RANK = 10;
    static final String EXECUTORCH_VERSION = "1.4.1"; // pinned ExecuTorch runtime version

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

    /**
     * DJL per-model option key controlling the XNNPACK workspace sharing mode, e.g. {@code
     * Criteria.optOption(EtEngine.WORKSPACE_SHARING_MODE_OPTION, "per_model")}. Accepted values are
     * {@code disabled}, {@code per_model}, or {@code global} (case-insensitive, trimmed). Precedence
     * is this option, then {@link #WORKSPACE_SHARING_MODE_PROPERTY}, then no spec at all -- leaving
     * the runtime's compiled-in default ({@code global} for our pin). An unrecognized option value
     * fails the model load; an unrecognized property value WARNs and is ignored.
     *
     * <p>Unlike {@link #NUM_THREADS_PROPERTY}, this is NOT process-global and NOT write-once:
     * ExecuTorch resolves the mode per delegate at load time, so modes compose across models and
     * load order does not matter.
     */
    public static final String WORKSPACE_SHARING_MODE_OPTION = EtWorkspaceSharing.OPTION_KEY;

    /**
     * JVM-wide default for models that do not carry {@link #WORKSPACE_SHARING_MODE_OPTION}, e.g.
     * {@code -Dai.djl.executorch.workspace_sharing_mode=disabled}. Accepted values are {@code
     * disabled}, {@code per_model}, or {@code global} (case-insensitive, trimmed). Precedence is
     * {@link #WORKSPACE_SHARING_MODE_OPTION}, then this property, then no spec at all -- leaving the
     * runtime's compiled-in default ({@code global} for our pin). An unrecognized property value
     * WARNs and is ignored (an unrecognized option value, by contrast, fails the model load).
     *
     * <p>Unlike {@link #NUM_THREADS_PROPERTY}, this is NOT process-global and NOT write-once:
     * ExecuTorch resolves the mode per delegate at load time, so modes compose across models and
     * load order does not matter.
     */
    public static final String WORKSPACE_SHARING_MODE_PROPERTY = EtWorkspaceSharing.PROPERTY;

    /**
     * DJL per-model option enabling ExecuTorch event tracing for one model, e.g. {@code
     * Criteria.optOption(EtEngine.PROFILING_OPTION, "true")}. Accepted values are {@code true} and
     * {@code false}; anything else fails the load.
     *
     * <p>There is no JVM-wide property counterpart by design — see {@code EtProfiling}.
     *
     * <p>Requires a runtime whose event tracer is compiled in; {@link #devtoolsAvailable()} reports
     * whether this platform has one, and requesting profiling without it fails the load.
     */
    public static final String PROFILING_OPTION = EtProfiling.OPTION_KEY;

    /**
     * JVM flag controlling whether the engine registers its JMX MBean, e.g.
     * {@code -Dai.djl.executorch.jmx_enabled=false}. Registration happens once, at the first model
     * load, under the object name {@value EtEngineStats#OBJECT_NAME}. Any value other than
     * {@code false} (case-insensitive) leaves it enabled.
     *
     * <p>Registration failure — a name collision, a {@code SecurityManager}, a restricted
     * container — is a single logged warning and never fails a model load.
     */
    public static final String JMX_ENABLED_PROPERTY = "ai.djl.executorch.jmx_enabled";

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
     *
     * @return the pool size the native runtime is actually using
     */
    public static int getIntraOpThreads() {
        return EtNative.intraOpThreads();
    }

    /**
     * Whether this platform's shipped native library can record ExecuTorch profiling data.
     *
     * @return true when the linked runtime has the event tracer compiled in
     */
    public static boolean devtoolsAvailable() {
        return EtNative.devtoolsAvailable();
    }

    /**
     * The numeric type OpenVINO will use for CPU inference on this host, e.g. {@code "f32"} or
     * {@code "bf16"}, or {@code "unavailable"} when it cannot be determined.
     *
     * <p>OpenVINO selects this from the CPU it lands on rather than from how the model was
     * compiled: on avx512_bf16/AMX hardware it computes in bf16, elsewhere in f32. Both are
     * correct, and the difference against an f32 golden is ~2.5e-3 versus ~6e-8 — which is why
     * OpenVINO parity tests use a loose tolerance. This exists so that looseness stays observable
     * instead of hiding a silent shift.
     *
     * <p>Reports what a <b>freshly created</b> Core would choose, not a reading from the Core the
     * delegate built. Those agree today because the choice derives from CPU capability alone.
     *
     * <p>Creating a Core loads the CPU plugin and is not cheap. This is an on-demand diagnostic:
     * do not call it on the hot path or during model load. It returns {@code "unavailable"} rather
     * than throwing, because a diagnostic that throws is a diagnostic people stop calling.
     *
     * @return the precision, or {@code "unavailable"}
     */
    public static String openVinoInferencePrecision() {
        String lib = OpenVinoRuntime.resolvedLibPath();
        if (lib == null) {
            return "unavailable";
        }
        try {
            // Null rather than a string means the native side could not even read its argument
            // (OOM pending). Fold it into the same sentinel: this is a diagnostic, and a caller
            // reading it should never have to distinguish degrees of unavailability.
            String precision = EtNative.openVinoInferencePrecision(lib);
            return (precision == null || precision.isEmpty()) ? "unavailable" : precision;
        } catch (RuntimeException | LinkageError e) {
            return "unavailable";
        }
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
            intraOpSealed = true;
            if (n >= 1) {
                int actual = EtNative.setIntraOpThreads(n);
                if (actual != n) {
                    logger.warn("{}: requested {} but the pool reports {}; using {}",
                            NUM_THREADS_PROPERTY, n, actual, actual);
                }
                sealedIntraOpThreads = actual;   // record the APPLIED count (issue #25)
                logger.info("intra-op thread pool sealed at {} ({}={})",
                        actual, NUM_THREADS_PROPERTY, n);
            } else {
                sealedIntraOpThreads = n;        // -1 = the runtime default
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
