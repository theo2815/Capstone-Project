package com.quickpitik.dto.ai

import com.fasterxml.jackson.annotation.JsonIgnoreProperties

// Minimal typed results for the endpoints the backend actually calls. Fields are added
// as downstream PRs consume them; unknown fields are ignored so the ai-api can evolve.

@JsonIgnoreProperties(ignoreUnknown = true)
data class BlurDetectResult(
    val is_blurry: Boolean = false,
    val confidence: Double = 0.0,
    val metrics: BlurMetrics = BlurMetrics(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class BlurMetrics(
    val laplacian_variance: Double = 0.0,
    val hf_ratio: Double = 0.0,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class FacesDetectResult(
    val faces: List<FaceDetection> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class FaceDetection(
    val confidence: Double = 0.0,
    val bbox: List<Double> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class FacesEnrollResult(
    val person_id: String,
    val faces_enrolled: Int = 1,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class FacesSearchResult(
    val matches: List<FaceMatch> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class FaceMatch(
    val person_id: String,
    val person_name: String? = null,
    val similarity: Double = 0.0,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class BibsRecognizeResult(
    val detections: List<BibDetection> = emptyList(),
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class BibDetection(
    val bib_number: String,
    val confidence: Double = 0.0,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class JobStatusResult(
    val job_id: String,
    val status: String,
    val progress: Double = 0.0,
    val result: List<Any?>? = null,
    val error: String? = null,
)

@JsonIgnoreProperties(ignoreUnknown = true)
data class HealthReady(
    val models_loaded: Boolean = false,
    val database: Boolean = false,
    val redis: Boolean = false,
)
