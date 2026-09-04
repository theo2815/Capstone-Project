package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// AWS Rekognition settings, used only when app.ai.provider=rekognition.
// Credentials are NOT here — they come from the AWS default chain
// (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env for dev, instance/task role in
// prod). Thresholds are on the 0–1 scale the backend already uses; the client
// converts to Rekognition's 0–100.
@ConfigurationProperties(prefix = "app.ai.rekognition")
data class RekognitionProperties(
    val region: String = "ap-southeast-1",
    // Optional static credentials. Leave unset to use the AWS default chain
    // (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env or ~/.aws/credentials for dev,
    // instance/task role in prod). Set them in application-local.yml to keep all
    // secrets in one place — mirrors S3StorageService.
    val accessKey: String? = null,
    val secretKey: String? = null,
    // Per-event collection id = prefix + eventId. One collection per event is the
    // tenant/isolation boundary (mirrors ai-api's event_id scoping).
    val collectionPrefix: String = "qp-event-",
    // Face-search match floor, 0–1 (sent to Rekognition as *100). 0.8 ≈ AWS's
    // recommended identity-match floor — the client never searches looser than
    // this even if a caller passes a smaller threshold.
    val faceMatchThreshold: Double = 0.8,
    // Max faces indexed per photo (bounds race-crowd photos).
    val maxFacesPerImage: Int = 15,
    // Minimum digit count for a DetectText token to be treated as a bib.
    val bibMinChars: Int = 2,
    // Downscale longest edge before the 5 MB inline call; only images already
    // over the inline-safe byte budget are decoded + re-encoded.
    val maxImageDimension: Int = 2048,
)
