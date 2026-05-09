package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable

@Embeddable
class PhotoFacePersonEmbed(
    @Column(name = "face_index", nullable = false)
    var faceIndex: Int,

    @Column(name = "ai_person_id", nullable = false, length = 100)
    var aiPersonId: String,
) {
    override fun equals(other: Any?): Boolean = this === other ||
        (other is PhotoFacePersonEmbed && faceIndex == other.faceIndex)

    override fun hashCode(): Int = faceIndex.hashCode()
}
