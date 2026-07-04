package org.measly.executorch.translate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.StringReader;
import java.util.List;
import org.junit.jupiter.api.Test;

class ModelSpecTest {
    @Test
    void parsesAndIgnoresUnknownFields() {
        String json =
                """
                { "runtime":"executorch", "extra":123, "inputs":[
                  {"name":"price","position":0,"dtype":"float32","shape":[1],"desc":"x"},
                  {"name":"qty","position":1,"dtype":"int64","shape":[1]}]}
                """;
        List<ParamSpec> specs = ModelSpec.parse(new StringReader(json));
        assertEquals(2, specs.size());
        assertEquals("price", specs.get(0).name());
        assertEquals(DType.FLOAT32, specs.get(0).dtype());
        assertEquals(1, specs.get(1).position());
        assertEquals(DType.INT64, specs.get(1).dtype());
    }

    @Test
    void sortsByPosition() {
        String json =
                """
                {"inputs":[
                  {"name":"b","position":1,"dtype":"int64","shape":[1]},
                  {"name":"a","position":0,"dtype":"float32","shape":[1]}]}
                """;
        List<ParamSpec> specs = ModelSpec.parse(new StringReader(json));
        assertEquals("a", specs.get(0).name());
        assertEquals("b", specs.get(1).name());
    }

    @Test
    void rejectsNonScalarShape() {
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"dtype\":\"float32\",\"shape\":[3]}]}")));
    }

    @Test
    void rejectsNegativeDimension() {
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"dtype\":\"float32\",\"shape\":[-1,-1]}]}")));
    }

    @Test
    void rejectsUnknownDtype() {
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"dtype\":\"bf16\",\"shape\":[1]}]}")));
    }

    @Test
    void rejectsEmptyInputs() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelSpec.parse(new StringReader("{\"inputs\":[]}")));
    }

    @Test
    void rejectsNullInputEntry() {
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[null]}")));
    }

    @Test
    void rejectsInputMissingRequiredField() {
        // dtype omitted -> Gson leaves it null -> IllegalArgumentException.
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(
                "{\"inputs\":[{\"name\":\"v\",\"position\":0,\"shape\":[1]}]}")));
    }

    @Test
    void rejectsDuplicatePosition() {
        String json =
                """
                {"inputs":[
                  {"name":"a","position":0,"dtype":"float32","shape":[1]},
                  {"name":"b","position":0,"dtype":"int64","shape":[1]}]}
                """;
        assertThrows(IllegalArgumentException.class, () -> ModelSpec.parse(new StringReader(json)));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(IllegalArgumentException.class,
                () -> ModelSpec.parse(new StringReader("{ this is not json ")));
    }
}
