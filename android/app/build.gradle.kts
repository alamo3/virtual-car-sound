plugins {
    id("com.android.application")
}

android {
    namespace = "com.virtualcarsound.remote"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.virtualcarsound.remote"
        minSdk = 26          // Android 8.0+
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")   // personal sideload build
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
