plugins {
    application
    alias(libs.plugins.jmh)
}

java {
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
}

repositories { mavenCentral() }

dependencies {
    implementation(project(":"))            // this ExecuTorch engine (brings its native .so via resources)
    implementation(libs.djl.pytorch.engine) // LibTorch baseline (auto-fetches native at runtime)
    implementation(libs.djl.api)            // Image, ImageClassificationTranslator
    runtimeOnly(libs.slf4j.api)
    runtimeOnly(libs.logback.classic)

    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

application {
    mainClass = "org.measly.example.MobilenetExample"
}

tasks.test { useJUnitPlatform() }

// Model artifacts are generated on demand into this directory (see the exportModels task, Task 3).
val modelsDir = layout.buildDirectory.dir("models")

//val exportModels by tasks.registering(Exec::class) {
val exportModels = tasks.register<Exec>("exportModels") {
    group = "build"
    description = "Generate MobileNetV2 .pte + .pt via uv (heavy; needs uv on PATH)."
    val out = modelsDir.get().asFile
    val script = rootProject.file("tools/scripts/export_mobilenet.py")
    inputs.file(script)
    outputs.files(
        out.resolve("mobilenet_v2.pte"),
        out.resolve("mobilenet_v2.pt"),
        out.resolve("versions.json"),
    )
    doFirst { out.mkdirs() }
    workingDir = out
    commandLine("uv", "run", script.absolutePath)
}

// Pass the models directory to the JVM so ModelArtifacts can resolve it at runtime.
tasks.named<JavaExec>("run") {
    systemProperty("example.models.dir", modelsDir.get().asFile.absolutePath)
}

// The plugin's standard fat jar writes META-INF/services/ai.djl.engine.EngineProvider
// twice — this project's ExecuTorch provider and djl-api's built-in RPC provider — as two
// separate zip entries. java.util.zip.ZipFile resolves duplicate names last-entry-wins.
// EXCLUDE keeps the first entry; project classes are added before the runtime
// classpath, so the surviving entry is the ExecuTorch provider.
tasks.named<Jar>("jmhJar") {
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

jmh {
    warmupIterations = 3
    iterations = 5
    fork = 1
    jvmArgs = listOf(
        "-Dexample.models.dir=" + modelsDir.get().asFile.absolutePath,
        // For this specific model, holding interop at 1 is ~11% faster than default on
        // my workstation (i7-1185G7)
        // ExecuTorch is not affected by this setting and in fact all threading settings are
        // set at compilation time
        "-Dai.djl.pytorch.num_interop_threads=1",
    )
}
