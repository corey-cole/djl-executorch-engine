plugins {
    `java-library`
    jacoco
    alias(libs.plugins.maven.publish)
    alias(libs.plugins.jacoco.to.cobertura)
}

group = "org.measly"
version = providers.gradleProperty("releaseVersion")
    .orElse(providers.environmentVariable("RELEASE_VERSION"))
    .getOrElse("0.1.0-SNAPSHOT")

java {
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
}

repositories { mavenCentral() }

dependencies {
    compileOnly(libs.djl.api)
    compileOnly(libs.slf4j.api)

    testImplementation(libs.djl.api)
    testImplementation(libs.slf4j.api)
    testImplementation(libs.logback.classic)
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.mockito.core)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform { excludeTags("leak", "oom", "openvino", "openvino-unsupported", "intraop", "jmx-disabled", "stats-degraded", "stress", "stress-sweep", "stress-baseline") }
    jvmArgs("-XX:+HeapDumpOnOutOfMemoryError")
    finalizedBy(tasks.jacocoTestReport)
}

// LibUtils resolves the native library from EXECUTORCH_LIBRARY_PATH before falling back to the
// classpath copy, so this variable changes WHICH .so/.dll is under test. Undeclared, it is invisible
// to the up-to-date check and the build cache (org.gradle.caching=true): point it at a different
// library and Gradle sees identical inputs, replays a cached result, and reports a pass for a run
// that loaded something else entirely. Declaring it makes the override part of the cache key.
// configureEach + withType so leakTest is covered too — it loads the same library.
tasks.withType<Test>().configureEach {
    inputs.property(
        "executorchLibraryPath",
        providers.environmentVariable("EXECUTORCH_LIBRARY_PATH").orElse("")
    )
    // -Xcheck:jni is the JVM's JNI-contract checker: JNI calls made with a pending exception, null
    // array arguments. It is on the umbrella rather than on tasks.test because tasks.test excludes
    // eight tags, and oomTest -- excluded -- is the task that drives the allocation-failure paths
    // this class of defect lives on. Costs nothing: it runs against the plain shipping library.
    jvmArgs("-Xcheck:jni")
}

tasks.register<Test>("leakTest") {
    description = "Memory-leak stress tests under constrained heap/direct memory."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("leak") }
    jvmArgs("-Xmx256m", "-XX:MaxDirectMemorySize=64m", "-XX:+HeapDumpOnOutOfMemoryError")
}

tasks.register<Test>("oomTest") {
    description = "JNI output-marshalling failure-contract tests under a constrained heap."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("oom") }
    jvmArgs("-Xmx128m")   // deliberately NO HeapDumpOnOutOfMemoryError: the OOM is the expected outcome
}

tasks.register<Test>("intraOpTest") {
    description = "Intra-op threadpool configuration tests; forked JVM because the pool is process-global."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("intraop") }
    jvmArgs("-Dai.djl.executorch.num_threads=2")
}

val jmxDisabledTest = tasks.register<Test>("jmxDisabledTest") {
    description = "Verifies ai.djl.executorch.jmx_enabled=false suppresses MBean registration."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("jmx-disabled") }
    jvmArgs("-Dai.djl.executorch.jmx_enabled=false")
}

val statsDegradedTest = tasks.register<Test>("statsDegradedTest") {
    description = "Verifies snapshot() degrades (never throws) when the native library is broken."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stats-degraded") }
    // Point the native load at a checked-in non-library file: any garbage file makes System.load
    // throw inside EtNative's <clinit>, which is exactly the degraded condition under test. The
    // value must be absolute — System.load needs a path, and the forked JVM's working directory is
    // not guaranteed to be the project root.
    environment(
        "EXECUTORCH_LIBRARY_PATH",
        file("src/test/resources/not-a-library.txt").absolutePath,
    )
}

// Forked per class: OPENVINO_LIB_PATH is process environment and the delegate's dlopen is
// once-only, so cases sharing a JVM contaminate each other in ways that present as flakes.
tasks.register<Test>("openvinoTest") {
  group = "verification"
  description = "OpenVINO delegate tests (linux-x86_64 with the openvino bundle)"
  testClassesDirs = sourceSets["test"].output.classesDirs
  classpath = sourceSets["test"].runtimeClasspath
  useJUnitPlatform { includeTags("openvino") }
  forkEvery = 1
}

// The inverse leg of openvinoTest: asserts the platform error where the delegate is ABSENT. Shaped
// exactly like openvinoTest (forked per class) so the two matrix legs stay symmetric.
tasks.register<Test>("openvinoUnsupportedTest") {
  group = "verification"
  description = "OpenVINO delegate tests (linux-x86_64 with the openvino bundle)"
  testClassesDirs = sourceSets["test"].output.classesDirs
  classpath = sourceSets["test"].runtimeClasspath
  useJUnitPlatform { includeTags("openvino-unsupported") }
  forkEvery = 1
}

val checkDocLinks = tasks.register<Exec>("checkDocLinks") {
    group = "verification"
    description = "Verifies every relative markdown link in tracked .md files resolves."
    commandLine("tools/scripts/check_doc_links.sh")
    // Skipped on Windows: Exec cannot run a .sh directly there, and the check is
    // platform-independent, so the Linux legs of CI already cover it for everyone.
    onlyIf { !System.getProperty("os.name").lowercase().contains("windows") }
}

// intraOpTest runs under build/check (its forked JVM cannot share the pool with the test task)
// but not under `test` itself, which excludes the tag. The opt-out and degraded-library branches
// are likewise forked-JVM-only, so they join check for parity.
tasks.check {
    dependsOn(
        tasks.named("intraOpTest"),
        jmxDisabledTest,
        statsDegradedTest,
        checkDocLinks,
    )
}

// --- Threading / workspace stress arms. All opt-in; `test` excludes every tag below. ---
// Never wire any of these to CI: they saturate every core for their whole duration, and free CI
// providers take a dim view of that. `stressGate` in particular is a local/self-hosted tool.

tasks.register<Test>("stressGate") {
    description = "Local-only concurrency correctness gate: 8 threads, maximum contention."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress") }
    // Deliberately NO num_threads property: the gate wants the real-world intra-op default, which
    // together with 8 caller threads is the maximum-contention configuration.
    systemProperty("et.stress.seconds", providers.gradleProperty("stressSeconds").getOrElse("30"))
}

// Run id coordination. The value must be FRESH per Gradle invocation even with the configuration
// cache (which freezes configuration-time values), so generation happens inside the forked test
// JVMs, coordinated through build/reports/stress/.run_id: stressSweepCore's doFirst clears the
// sidecar (or primes it with -PstressRunId) at execution time, the core JVM writes its fresh id
// there, and the baseline JVM (which runs after, mustRunAfter) adopts it. The system property
// below additionally honors -PstressRunId for any path where the sidecar is absent.

// The sweep is split across two JVMs because the intra-op pool is process-global and write-once:
// measly::et::setIntraOpThreads refuses a reset once any EtRuntime exists (issue #26), so the eight
// intra-op=1 cells and the intra-op=default confirmation cell cannot share a process.
tasks.register<Test>("stressSweepCore") {
    description = "Throughput sweep: {1,2,4,8} threads x {global,disabled} at ONE intra-op thread."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress-sweep") }
    jvmArgs("-Dai.djl.executorch.num_threads=1")
    // Locals (not script properties) so the doFirst closure is configuration-cache serializable.
    val stressRunId = providers.gradleProperty("stressRunId").getOrElse("")
    val idFile = layout.buildDirectory.file("reports/stress/.run_id").get().asFile
    systemProperty("et.stress.runId", stressRunId)
    systemProperty("et.stress.cellSeconds", providers.gradleProperty("stressCellSeconds").getOrElse("10"))
    doFirst {
        // Execution-time so the per-invocation clear cannot be frozen; both captured values
        // (String, File) are configuration-cache serializable.
        if (stressRunId.isEmpty()) {
            idFile.delete()
        } else {
            idFile.parentFile.mkdirs()
            idFile.writeText(stressRunId)
        }
    }
}

tasks.register<Test>("stressSweepBaseline") {
    description = "Sweep confirmation cell: 1 thread at the DEFAULT intra-op pool size."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("stress-baseline") }
    mustRunAfter(tasks.named("stressSweepCore")) // both append to one report; keep the order stable
    val stressRunId = providers.gradleProperty("stressRunId").getOrElse("")
    val idFile = layout.buildDirectory.file("reports/stress/.run_id").get().asFile
    systemProperty("et.stress.runId", stressRunId)
    systemProperty("et.stress.cellSeconds", providers.gradleProperty("stressCellSeconds").getOrElse("10"))
    doFirst {
        // Adopt the override if given (e.g. standalone baseline run); never clear the sidecar —
        // the core arm wrote this invocation's id there and the baseline must match it.
        if (stressRunId.isNotEmpty()) {
            idFile.parentFile.mkdirs()
            idFile.writeText(stressRunId)
        }
    }
}

tasks.register("stressSweep") {
    description = "Full 9-cell sweep (stressSweepCore + stressSweepBaseline)."
    group = "verification"
    dependsOn(tasks.named("stressSweepCore"), tasks.named("stressSweepBaseline"))
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required = true
        csv.required = false
        html.required = true
    }
}

val nativePlatforms = listOf("linux-x86_64", "linux-aarch64", "windows-x86_64")

// MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with LibUtils.libName.
fun nativeLibName(platform: String): String =
    if (platform.startsWith("windows-")) "executorch_djl.dll" else "libexecutorch_djl.so"

// Look for the platform's native library in build/native-staging/<platform>/<nativeLibName>
val nativeStaging = layout.buildDirectory.dir("native-staging")
val nativeJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}") {
    archiveClassifier.set(platform)
    // The native library, excluding the bundled licenses subtree (mapped to META-INF below) and
    // the OpenVINO runtime bundle (shipped only via its own opt-in openvino variant).
    from(nativeStaging.map { it.dir(platform) }) {
        exclude("licenses/**")
        exclude("openvino/**")
        into("native/${platform}")
    }
    // Third-party notices from the runtime tarball, staged next to the .so by native/build.sh.
    from(nativeStaging.map { it.dir("${platform}/licenses") }) {
        into("META-INF/licenses/executorch-runtime")
    }
    // Resolve to plain Files at configuration time so doFirst captures only File + String
    // (config-cache safe) rather than the enclosing script.
    val resolvedLib = nativeStaging.get().dir(platform).file(nativeLibName(platform)).asFile
    val licensesDir = nativeStaging.get().dir(platform).dir("licenses").asFile
    doFirst { // Fail a release rather than ship an empty native jar or a binary with no notices
        require(resolvedLib.exists()) { "Missing native library for ${platform}: ${resolvedLib}" }
        require(licensesDir.isDirectory && (licensesDir.list()?.isNotEmpty() ?: false)) {
            "Missing third-party notices for ${platform}: ${licensesDir}" +
                " (native/build.sh must stage LICENSE + THIRD-PARTY-NOTICES/)"
        }
    }
  }
}

// The OpenVINO runtime ships as a SEPARATE opt-in variant, never folded into the platform jar:
// it is ~21 MB compressed and ~72 MB extracted, for a delegate most consumers never load. A
// consumer opts in by requesting the capability.
//
// Registered for every platform but only produced where build.sh staged a bundle -- the pin decides
// which platforms have one, so this needs no platform name.
val openvinoJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}-openvino") {
    archiveClassifier.set("${platform}-openvino")
    from(nativeStaging.map { it.dir("${platform}/openvino") }) {
      exclude("licenses/**")
      into("native/${platform}/openvino")
    }
    from(nativeStaging.map { it.dir("${platform}/openvino/licenses") }) {
      into("META-INF/licenses/openvino-runtime")
    }
    val bundleDir = nativeStaging.get().dir(platform).dir("openvino").asFile
    onlyIf { bundleDir.isDirectory }
    doFirst { // A jar with the libraries but no notices is not shippable
      require(File(bundleDir, "licenses").isDirectory) {
        "Missing OpenVINO third-party notices for ${platform}: ${bundleDir}/licenses"
      }
    }
  }
}

// Publish each native jar as a real GMM variant (not just a bare classified artifact) so Gradle
// consumers can select it via attributes/capability instead of a Maven classifier string. The
// per-platform capability keeps the variants out of default resolution: without it, both native
// variants match the standard JVM attribute set and an attribute-less consumer hits ambiguity.
val nativeVariants = nativePlatforms.map { platform ->
    val osFamily = platform.substringBefore("-") // linux / windows
    val arch = platform.substringAfter("-") // x86_64 / aarch64
    configurations.consumable("nativeRuntimeElements-${platform}") {
        attributes {
            attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
            attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
            attribute(Bundling.BUNDLING_ATTRIBUTE, objects.named(Bundling.EXTERNAL))
            attribute(LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE, objects.named(LibraryElements.JAR))
            attribute(OperatingSystemFamily.OPERATING_SYSTEM_ATTRIBUTE, objects.named(osFamily))
            attribute(
                MachineArchitecture.ARCHITECTURE_ATTRIBUTE,
                objects.named(if (arch == "aarch64") MachineArchitecture.ARM64 else MachineArchitecture.X86_64)
            )
        }
        outgoing {
            capability("${project.group}:djl-executorch-engine-${platform}:${project.version}")
            artifact(tasks.named("nativeJar-${platform}"))
        }
    }
}

val openvinoVariants = nativePlatforms.map { platform ->
    val osFamily = platform.substringBefore("-")
    val arch = platform.substringAfter("-")
    configurations.consumable("openvinoRuntimeElements-${platform}") {
        attributes {
            attribute(Category.CATEGORY_ATTRIBUTE, objects.named(Category.LIBRARY))
            attribute(Usage.USAGE_ATTRIBUTE, objects.named(Usage.JAVA_RUNTIME))
            attribute(Bundling.BUNDLING_ATTRIBUTE, objects.named(Bundling.EXTERNAL))
            attribute(LibraryElements.LIBRARY_ELEMENTS_ATTRIBUTE, objects.named(LibraryElements.JAR))
            attribute(OperatingSystemFamily.OPERATING_SYSTEM_ATTRIBUTE, objects.named(osFamily))
            attribute(
                MachineArchitecture.ARCHITECTURE_ATTRIBUTE,
                objects.named(if (arch == "aarch64") MachineArchitecture.ARM64 else MachineArchitecture.X86_64)
            )
        }
        outgoing {
            capability("${project.group}:djl-executorch-engine-${platform}-openvino:${project.version}")
            artifact(tasks.named("nativeJar-${platform}-openvino"))
        }
    }
}

(components["java"] as AdhocComponentWithVariants).apply {
    nativeVariants.forEach { variant ->
        addVariantsFromConfiguration(variant.get()) { mapToOptional() }
    }
    openvinoVariants.forEach { variant ->
        addVariantsFromConfiguration(variant.get()) { mapToOptional() }
    }
}

mavenPublishing {
    //publishToMavenCentral(automaticRelease = true, validateDeployment = DeploymentValidation.PUBLISHED)
    publishToMavenCentral()
    signAllPublications()

    coordinates(group.toString(), "djl-executorch-engine", version.toString())

    pom {
        name.set("DJL ExecuTorch Engine")
        description.set("Enable use of ExecuTorch (PTE) models within DJL")
        inceptionYear.set("2026")
        url.set("https://github.com/corey-cole/djl-executorch-engine")
        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
            }
        }
        developers {
            developer {
                id.set("corey-cole")
                name.set("Corey Cole")
                url.set("https://github.com/corey-cole")
            }
        }
        scm {
            url.set("https://github.com/corey-cole/djl-executorch-engine")
            connection.set("scm:git:git://github.com/corey-cole/djl-executorch-engine.git")
            developerConnection.set("scm:git:ssh://git@github.com/corey-cole/djl-executorch-engine.git")
        }
    }
}
