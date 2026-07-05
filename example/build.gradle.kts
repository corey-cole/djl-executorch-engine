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
}

application {
    mainClass = "org.measly.example.MobilenetExample"
}

// Model artifacts are generated on demand into this directory (see the exportModels task, Task 3).
val modelsDir = layout.buildDirectory.dir("models")

// Pass the models directory to the JVM so ModelArtifacts can resolve it at runtime.
tasks.named<JavaExec>("run") {
    systemProperty("example.models.dir", modelsDir.get().asFile.absolutePath)
}
