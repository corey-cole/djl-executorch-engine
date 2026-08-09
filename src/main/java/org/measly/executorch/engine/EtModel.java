package org.measly.executorch.engine;

import ai.djl.BaseModel;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.measly.executorch.jni.EtMethodMeta;
import org.measly.executorch.jni.EtNative;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** ExecuTorch {@link ai.djl.Model}: loads a .pte and owns its native handle via the block. */
public class EtModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(EtModel.class);

    private static final String PTE_SUFFIX = ".pte";

    EtModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("etModel");
        dataType = DataType.FLOAT32;
    }

    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws IOException, MalformedModelException {
        setModelDir(modelPath);
        if (block != null) {
            throw new UnsupportedOperationException("ExecuTorch does not support dynamic blocks");
        }
        Path modelFile;
        if (prefix != null) {
            modelFile = findPteFile(prefix);
        } else {
            modelFile = findPteFile(modelName, modelDir.toFile().getName(), "model.pte");
        }
        if (modelFile == null) {
            throw new FileNotFoundException(".pte file not found in: " + modelPath);
        }
        // Per-model, resolved fresh on every load: nothing is sealed, and load order does not
        // matter. Throws IllegalArgumentException for a bad per-model option. Resolved before the
        // intra-op seal below so a bad option here fails fast, before anything irreversible happens.
        int workspaceSharingMode =
                EtWorkspaceSharing.resolve(options, System.getProperty(EtWorkspaceSharing.PROPERTY));
        // First load seals the process-global intra-op thread pool (applies pending/property value,
        // logs the outcome); later loads are no-ops. Must precede loadModule: delegate init during
        // load submits work to the pool.
        EtEngine.sealIntraOpThreads();
        logger.info(
                "model {} workspaceSharingMode={}", getName(), EtWorkspaceSharing.name(workspaceSharingMode));
        // Not unit-tested below this point: loadModule/methodMeta/destroy require the native library
        // (integration-tested via EtModelTest#loadAndForwardAddModel).
        // Timed from here so loadNanos covers delegate initialisation: EtRuntime's constructor
        // calls Module::load_forward() unconditionally, so the XNNPACK setup cost lands in load,
        // not in the first forward.
        final long loadStartNanos = System.nanoTime();
        long handle = EtNative.loadModule(modelFile.toString(), workspaceSharingMode);
        EtMethodMeta meta;
        try {
            meta = EtNative.methodMeta(handle);
        } catch (RuntimeException e) {
            EtNative.destroy(handle); // don't leak the native module if metadata query fails
            throw e;
        }
        final long loadNanos = System.nanoTime() - loadStartNanos;
        EtSymbolBlock etBlock = new EtSymbolBlock(handle, (EtNDManager) manager, meta);
        etBlock.attachCounters(
                new EtModelCounters(
                        getName(),
                        EtWorkspaceSharing.name(workspaceSharingMode),
                        meta.plannedArenaBytes,
                        loadNanos));
        block = etBlock;
        EtEngineStats.register(handle, etBlock);
        // After registration so the first JMX read already sees this model. One-shot: later loads
        // return immediately.
        EtEngineStats.registerMBeanOnce();
        for (int i = 0; i < meta.numInputs; i++) {
            logger.info("model {} input {} memoryPlanned={}", getName(), i, meta.inputMemoryPlanned[i]);
        }
        wasLoaded = true;
    }

    @Override
    public void close() {
        if (block instanceof EtSymbolBlock) {
            ((EtSymbolBlock) block).close();
        }
        super.close();
    }

    /**
     * Searches {@code modelDir} for a .pte file matching any of the given prefixes.
     * Mirrors the private findModelFile logic in OrtModel, adapted for .pte.
     */
    private Path findPteFile(String... prefixes) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            String fileName = file.toFile().getName();
            if (fileName.endsWith(PTE_SUFFIX)) {
                modelName = fileName.substring(0, fileName.length() - 4);
            } else {
                modelName = fileName;
            }
            return file;
        }
        for (String prefix : prefixes) {
            Path candidate = modelDir.resolve(prefix);
            if (Files.isRegularFile(candidate)) {
                return candidate;
            }
            if (!prefix.endsWith(PTE_SUFFIX)) {
                candidate = modelDir.resolve(prefix + PTE_SUFFIX);
                if (Files.isRegularFile(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }
}
