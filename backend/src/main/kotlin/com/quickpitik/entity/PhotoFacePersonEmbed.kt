package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import jakarta.persistence.Enumerated
import jakarta.persistence.EnumType

enum class FaceProvider { UNKNOWN, AI_API, REKOGNITION }

@Embeddable
class PhotoFacePersonEmbed(
    @Column(name = "face_index", nullable = false)
    var faceIndex: Int,

    @Column(name = "ai_person_id", nullable = false, length = 100)
    var aiPersonId: String,

    @Enumerated(EnumType.STRING)
    @Column(name = "provider", nullable = false, length = 16)
    var provider: FaceProvider = FaceProvider.UNKNOWN,
) {
    override fun equals(other: Any?): Boolean = this === other ||
        (other is PhotoFacePersonEmbed && faceIndex == other.faceIndex)

    override fun hashCode(): Int = faceIndex.hashCode()
}
