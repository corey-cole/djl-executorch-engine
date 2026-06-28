package org.measly.executorch.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

/** ExecuTorch implementation of {@link Engine}. CPU-only, limited NDArray support. */
public final class EtEngine extends Engine {

    public static final String ENGINE_NAME = "ExecuTorch";
    static final int RANK = 10;

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
        return "1.3.1"; // pinned ExecuTorch runtime version
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

    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
