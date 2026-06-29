package org.measly.executorch.translate;

import ai.djl.util.JsonUtils;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Parses model_spec.json (lenient: unknown fields ignored) into position-sorted ParamSpecs.
 *
 * <p>Positions need not be contiguous or 0-based; only relative order matters (the result is sorted
 * by position). Any malformed or invalid spec surfaces as {@link IllegalArgumentException}.
 */
final class ModelSpec {

    private ModelSpec() {}

    static List<ParamSpec> parse(Path jsonFile) throws IOException {
        try (Reader r = Files.newBufferedReader(jsonFile)) {
            return parse(r);
        }
    }

    /** Parses from {@code reader}; the caller retains ownership of the reader (it is not closed here). */
    static List<ParamSpec> parse(Reader reader) {
        Dto dto;
        try {
            dto = JsonUtils.GSON.fromJson(reader, Dto.class);
        } catch (com.google.gson.JsonSyntaxException e) {
            throw new IllegalArgumentException("Malformed model_spec.json", e);
        }
        if (dto == null || dto.inputs == null || dto.inputs.isEmpty()) {
            throw new IllegalArgumentException("model_spec.json has no inputs");
        }
        List<ParamSpec> specs = new ArrayList<>(dto.inputs.size());
        Set<Integer> positions = new HashSet<>();
        for (InputDto in : dto.inputs) {
            if (in == null) {
                throw new IllegalArgumentException("model_spec.json contains a null input entry");
            }
            if (in.name == null || in.dtype == null || in.shape == null) {
                throw new IllegalArgumentException("model_spec.json input missing name/dtype/shape");
            }
            if (!positions.add(in.position)) {
                throw new IllegalArgumentException("Duplicate input position " + in.position);
            }
            long product = 1;
            for (long d : in.shape) {
                if (d < 0) {
                    throw new IllegalArgumentException(
                            "Negative dimension in shape of '" + in.name + "'");
                }
                product *= d;
            }
            // Empty shape [] (a 0-d tensor) has product 1 and is an accepted scalar, as is [1].
            if (product != 1) {
                throw new IllegalArgumentException(
                        "Phase 2b supports scalar params only; '" + in.name + "' has shape "
                                + Arrays.toString(in.shape));
            }
            specs.add(new ParamSpec(in.name, in.position, DType.from(in.dtype), in.shape));
        }
        specs.sort(Comparator.comparingInt(ParamSpec::position));
        return specs;
    }

    /** Gson DTO matching the JSON; unknown fields are ignored by Gson. */
    private static final class Dto {
        List<InputDto> inputs;
    }

    // Q: Could this be a record?
    private static final class InputDto {
        String name;
        int position;
        String dtype;
        long[] shape;
    }
}
