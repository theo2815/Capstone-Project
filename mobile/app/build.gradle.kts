import java.util.Properties

plugins {
    alias(libs.plugins.androidApplication)
    alias(libs.plugins.jetbrainsKotlinAndroid)
    kotlin("kapt")
}

// Fujifilm SDK capture is opt-in even on machines that copied the proprietary files.
val fujiSdkDir = rootProject.file("fuji-sdk")
val fujiSdk = providers.gradleProperty("QP_ENABLE_FUJI_SDK").getOrElse("false").toBooleanStrictOrNull()
    ?: throw GradleException("QP_ENABLE_FUJI_SDK must be true or false.")
if (fujiSdk) {
    val missing = listOf("HEADERS/XAPI.H", "Android/arm64-v8a/XAPI.so")
        .filterNot { fujiSdkDir.resolve(it).isFile }
    if (missing.isNotEmpty()) throw GradleException(
        "QP_ENABLE_FUJI_SDK=true requires these files under mobile/fuji-sdk: ${missing.joinToString()}"
    )
}

android {
    namespace = "com.quickpitik.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.quickpitik.mobile"
        minSdk = 29
        targetSdk = 34
        versionCode = 2
        versionName = "1.0"

        vectorDrawables {
            useSupportLibrary = true
        }

        // Google OAuth web client ID from gradle.properties — see the note
        // there. Compiled in (not runtime-settable) so release builds can't be
        // repointed, same stance as DEFAULT_BASE_URL.
        buildConfigField(
            "String",
            "GOOGLE_SERVER_CLIENT_ID",
            "\"${project.findProperty("QP_GOOGLE_SERVER_CLIENT_ID") ?: ""}\"",
        )
        buildConfigField("boolean", "FUJI_SDK", fujiSdk.toString())
        if (fujiSdk) {
            ndk { abiFilters += "arm64-v8a" }
            externalNativeBuild { cmake { arguments += "-DFUJI_SDK_DIR=${fujiSdkDir.absoluteFile.invariantSeparatorsPath}" } }
        }
    }

    if (fujiSdk) {
        ndkVersion = "27.2.12479018"
        externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt") } }
        sourceSets["main"].jniLibs.srcDir(fujiSdkDir.resolve("Android"))
    }

    // Release signing from a gitignored mobile/keystore.properties:
    //   storeFile=<path to .jks>  storePassword=…  keyAlias=…  keyPassword=…
    // Absent file → the release APK is built unsigned (uninstallable), so debug
    // builds and CI without the keystore still work. The keystore's SHA-1 must
    // also be registered as an Android OAuth client in Google Cloud.
    val keystoreProps = rootProject.file("keystore.properties")
    if (keystoreProps.exists()) {
        val props = Properties().apply { keystoreProps.inputStream().use { load(it) } }
        signingConfigs {
            create("release") {
                storeFile = rootProject.file(props.getProperty("storeFile"))
                storePassword = props.getProperty("storePassword")
                keyAlias = props.getProperty("keyAlias")
                keyPassword = props.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        debug {
            manifestPlaceholders["usesCleartextTraffic"] = "true"
        }
        release {
            isMinifyEnabled = false
            manifestPlaceholders["usesCleartextTraffic"] = "false"
            signingConfig = signingConfigs.findByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        compose = true
        // AGP 8 stopped generating BuildConfig unless asked. RetrofitClient and
        // the Login screen's debug server field both gate on BuildConfig.DEBUG
        // so no release build can be pointed at another backend.
        buildConfig = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.14"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
    testOptions {
        unitTests {
            // Robolectric needs the merged manifest + resources on the JVM
            // test classpath; without this it can't inflate an Application.
            isIncludeAndroidResources = true
        }
    }
}

kapt {
    arguments { arg("room.schemaLocation", "$projectDir/schemas") }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.security.crypto)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    // Downloadable Google Fonts (Bricolage Grotesque / Funnel Sans / Geist Mono — website parity)
    implementation(libs.androidx.ui.text.google.fonts)

    // Networking
    implementation(libs.retrofit)
    implementation(libs.retrofit.converter.gson)
    implementation(libs.okhttp)
    implementation(libs.okhttp.logging.interceptor)

    // SQLite Persistence (Room)
    implementation(libs.room.runtime)
    implementation(libs.room.ktx)
    kapt(libs.room.compiler)

    // Background Sync (WorkManager)
    implementation(libs.work.runtime.ktx)

    // Coil Image Loader
    implementation(libs.coil.compose)

    // Jetpack Compose Navigation
    implementation(libs.navigation.compose)

    // Custom Tabs — keeps the PayMongo handoff out of an app-chooser dialog
    implementation(libs.browser)

    // "Continue with Google" — Credential Manager + Google ID token parsing
    implementation(libs.androidx.credentials)
    implementation(libs.androidx.credentials.play.services.auth)
    implementation(libs.googleid)

    testImplementation(libs.junit)
    testImplementation(libs.room.testing)
    testImplementation(libs.work.testing)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.ui.test.junit4)
    testImplementation(libs.mockwebserver)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
}
