plugins {
	kotlin("jvm") version "1.9.25"
	kotlin("plugin.spring") version "1.9.25"
	kotlin("plugin.jpa") version "1.9.25"
	id("org.springframework.boot") version "3.5.14"
	id("io.spring.dependency-management") version "1.1.7"
}

group = "com.quickpitik"
version = "0.0.1-SNAPSHOT"

java {
	toolchain {
		languageVersion = JavaLanguageVersion.of(21)
	}
}

repositories {
	mavenCentral()
}

dependencies {
	implementation("org.springframework.boot:spring-boot-starter-web")
	implementation("org.springframework.boot:spring-boot-starter-security")
	implementation("org.springframework.boot:spring-boot-starter-data-jpa")
	implementation("org.springframework.boot:spring-boot-starter-validation")
	implementation("org.springframework.boot:spring-boot-starter-websocket")
	implementation("com.fasterxml.jackson.module:jackson-module-kotlin")
	implementation("com.fasterxml.jackson.datatype:jackson-datatype-jsr310")
	implementation("org.flywaydb:flyway-core")
	implementation("org.flywaydb:flyway-database-postgresql")
	implementation("org.jetbrains.kotlin:kotlin-reflect")

	implementation("io.jsonwebtoken:jjwt-api:0.12.6")
	runtimeOnly("io.jsonwebtoken:jjwt-impl:0.12.6")
	runtimeOnly("io.jsonwebtoken:jjwt-jackson:0.12.6")

	implementation("software.amazon.awssdk:s3:2.30.21")

	// Pooled HTTP client for the ai-api RestClient (RestClientConfig). Async
	// photo indexing fans out concurrent face + bib calls; the default
	// SimpleClientHttpRequestFactory opens one connection per request. Version
	// managed by the Spring Boot BOM.
	implementation("org.apache.httpcomponents.client5:httpclient5")

	// EXIF orientation reader. Phone cameras (iPhone especially) tag portrait
	// photos with an orientation flag instead of rotating the pixel buffer.
	// Java's ImageIO doesn't honor the tag — without this, every portrait
	// upload renders sideways. Used by WatermarkService.processThumbnail.
	implementation("com.drewnoakes:metadata-extractor:2.19.0")

	// In-memory rate limiter (single-instance v1; Redis-backed swap is a future
	// PR per docs/IMPLEMENTATION_PLAN.md scaling notes).
	implementation("com.bucket4j:bucket4j-core:8.10.1")

	// OpenAPI 3 spec + Swagger UI, generated from the controllers. Not in the
	// Spring Boot BOM, so the version is explicit; the 2.8.x line targets Boot
	// 3.4/3.5. Served only when API_DOCS_ENABLED=true. See OpenApiConfig for
	// why the generated schemas have to be post-processed.
	implementation("org.springdoc:springdoc-openapi-starter-webmvc-ui:2.8.9")

	runtimeOnly("org.postgresql:postgresql")

	testImplementation("org.springframework.boot:spring-boot-starter-test")
	testImplementation("org.springframework.security:spring-security-test")
	testImplementation("org.jetbrains.kotlin:kotlin-test-junit5")
	testRuntimeOnly("org.junit.platform:junit-platform-launcher")

	// Real-Postgres integration tests, tagged "integration" and excluded from
	// `test` — see the task wiring below. Versions come from the Spring Boot
	// BOM's testcontainers dependency management.
	testImplementation("org.springframework.boot:spring-boot-testcontainers")
	testImplementation("org.testcontainers:junit-jupiter")
	testImplementation("org.testcontainers:postgresql")
}

kotlin {
	compilerOptions {
		freeCompilerArgs.addAll("-Xjsr305=strict")
	}
}

allOpen {
	annotation("jakarta.persistence.Entity")
	annotation("jakarta.persistence.MappedSuperclass")
	annotation("jakarta.persistence.Embeddable")
}

// `./gradlew test` stays Docker-free — backend/CLAUDE.md promises that, and it
// is what keeps the unit suite runnable on a machine with nothing installed but
// a JDK. Everything that needs a real Postgres is tagged "integration" and runs
// only under the `integrationTest` task below.
tasks.test {
	useJUnitPlatform {
		excludeTags("integration")
	}
}

// Spins a throwaway Postgres 16 per run via Testcontainers; requires a running
// Docker daemon. Shares the `test` source set rather than adding a second one —
// three test classes do not justify the extra Gradle wiring.
val integrationTest by tasks.registering(Test::class) {
	description = "Runs @Tag(\"integration\") tests against a Testcontainers Postgres."
	group = "verification"
	useJUnitPlatform {
		includeTags("integration")
	}
	testClassesDirs = sourceSets.test.get().output.classesDirs
	classpath = sourceSets.test.get().runtimeClasspath
	shouldRunAfter(tasks.test)
}
