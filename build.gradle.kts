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

val nativePlatforms = listOf("linux-x86_64")
// Look for .so files in build/native-staging/<platform>/libexecutorch_djl.so
val nativeStaging = layout.buildDirectory.dir("native-staging")
val nativeJarTasks = nativePlatforms.map { platform ->
  tasks.register<Jar>("nativeJar-${platform}") {
    archiveClassifier.set(platform)
    from(nativeStaging.map { it.dir(platform) }) {
        into("native/${platform}")
    }
    // Resolve to a plain File at configuration time so the doFirst action captures
    // only a File + String (config-cache safe) rather than the enclosing script.
    val resolvedSo = nativeStaging.get().dir(platform).file("libexecutorch_djl.so").asFile
    doFirst { // Fail a release rather than ship an empty native jar
        require(resolvedSo.exists()) { "Missing native library for ${platform}: ${resolvedSo}" }
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
