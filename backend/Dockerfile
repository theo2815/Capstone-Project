# Railway build for the Spring Boot backend (service Root Directory = backend/).
# Two stages: JDK 21 to build the boot jar, JRE 21 + fontconfig to run it.
# fontconfig is what lets WatermarkService rasterise the credit caption — without
# it previews silently ship with wordmark tiles only (see backend/CLAUDE.md).

FROM eclipse-temurin:21-jdk AS build
WORKDIR /src
COPY gradlew build.gradle.kts settings.gradle.kts ./
COPY gradle ./gradle
RUN chmod +x gradlew && ./gradlew --no-daemon dependencies -q > /dev/null 2>&1 || true
COPY src ./src
RUN ./gradlew --no-daemon clean bootJar -x test

FROM eclipse-temurin:21-jre
RUN apt-get update \
    && apt-get install -y --no-install-recommends fontconfig \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/libs/backend-*.jar app.jar
# Heap sized to the container, not the JVM's 25% default; AWT stays headless.
ENV JAVA_TOOL_OPTIONS="-XX:MaxRAMPercentage=70 -Djava.awt.headless=true"
EXPOSE 8080
# Railway injects PORT; honour it directly so SERVER_PORT=${{PORT}} is optional.
ENTRYPOINT ["sh", "-c", "exec java -jar /app/app.jar --server.port=${PORT:-${SERVER_PORT:-8080}}"]
