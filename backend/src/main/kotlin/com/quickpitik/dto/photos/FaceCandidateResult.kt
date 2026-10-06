package com.quickpitik.dto.photos

data class FaceCandidate(val personId: String, val similarity: Double)
data class FaceCandidateResult(
    val matches: List<FaceCandidate>,
    val candidateLimit: Int,
    val possiblyTruncated: Boolean,
    val noFaces: Boolean = false,
)
