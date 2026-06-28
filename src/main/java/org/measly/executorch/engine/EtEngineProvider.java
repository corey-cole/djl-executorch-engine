package org.measly.executorch.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** ExecuTorch implementation of {@link EngineProvider}. */
public final class EtEngineProvider implements EngineProvider {

    @Override
    public String getEngineName() {
        return EtEngine.ENGINE_NAME;
    }

    @Override
    public int getEngineRank() {
        return EtEngine.RANK;
    }

    @Override
    public Engine getEngine() {
        return InstanceHolder.INSTANCE;
    }

    private static final class InstanceHolder {
        static final Engine INSTANCE = EtEngine.newInstance();
    }
}
