plugins {
    `java-library`
    id("com.vanniktech.maven.publish") version "0.37.0"
}

group = "org.measly"
version = "0.1.0-SNAPSHOT"

java {
    toolchain { languageVersion = JavaLanguageVersion.of(17) }
}

repositories { mavenCentral() }

val djlVersion = "0.36.0"

dependencies {
    compileOnly("ai.djl:api:$djlVersion")
    compileOnly("org.slf4j:slf4j-api:2.0.17")

    testImplementation("ai.djl:api:$djlVersion")
    testImplementation("org.slf4j:slf4j-api:2.0.17")
    testImplementation("ch.qos.logback:logback-classic:1.5.18")
    testImplementation(platform("org.junit:junit-bom:5.10.2"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform { excludeTags("leak") }
    jvmArgs("-XX:+HeapDumpOnOutOfMemoryError")
}

tasks.register<Test>("leakTest") {
    description = "Memory-leak stress tests under constrained heap/direct memory."
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("leak") }
    jvmArgs("-Xmx256m", "-XX:MaxDirectMemorySize=64m", "-XX:+HeapDumpOnOutOfMemoryError")
}

mavenPublishing {
    // Automatically generates sources, javadocs, and handles GPG signing
    publishToMavenCentral()
    
    // Automatically signs if GPG environment variables are present
    signAllPublications()

    coordinates("org.measly", "djl-executorch-engine", "0.1.0-SNAPSHOT")

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
