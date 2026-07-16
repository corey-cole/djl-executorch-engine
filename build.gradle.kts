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
    useJUnitPlatform { excludeTags("leak") }
    jvmArgs("-XX:+HeapDumpOnOutOfMemoryError")
    finalizedBy(tasks.jacocoTestReport)
}

tasks.register<Test>("leakTest") {
    description = "Memory-leak stress tests under constrained heap/direct memory."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("leak") }
    jvmArgs("-Xmx256m", "-XX:MaxDirectMemorySize=64m", "-XX:+HeapDumpOnOutOfMemoryError")
}

tasks.jacocoTestReport {
    dependsOn(tasks.test)
    reports {
        xml.required = true
        csv.required = false
        html.required = true
    }
}

val nativePlatforms = listOf("linux-x86_64", "windows-x86_64")

// MSVC emits no `lib` prefix and a .dll suffix. Keep in sync with LibUtils.libName.
fun nativeLibName(platform: String): String =
    if (platform.startsWith("windows-")) "executorch_djl.dll" else "libexecutorch_djl.so"

// Look for the platform's native library in build/native-staging/<platform>/<nativeLibName>
val nativeStaging = layout.buildDirectory.dir("native-staging")
val nativeJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}") {
    archiveClassifier.set(platform)
    // The native library, excluding the bundled licenses subtree (mapped to META-INF below).
    from(nativeStaging.map { it.dir(platform) }) {
        exclude("licenses/**")
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

publishing {
    publications.withType<MavenPublication>().configureEach {
        nativeJarTasks.forEach { artifact(it) }
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
