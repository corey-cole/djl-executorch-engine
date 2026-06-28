package org.measly.executorch.engine;

import ai.djl.BaseModel;
import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.measly.executorch.jni.EtNative;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** ExecuTorch {@link ai.djl.Model}: loads a .pte and owns its native handle via the block. */
public class EtModel extends BaseModel {

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
        long handle = EtNative.loadModule(modelFile.toString());
        block = new EtSymbolBlock(handle, (EtNDManager) manager);
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
            if (fileName.endsWith(".pte")) {
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
            if (!prefix.endsWith(".pte")) {
                candidate = modelDir.resolve(prefix + ".pte");
                if (Files.isRegularFile(candidate)) {
                    return candidate;
                }
            }
        }
        return null;
    }
}
