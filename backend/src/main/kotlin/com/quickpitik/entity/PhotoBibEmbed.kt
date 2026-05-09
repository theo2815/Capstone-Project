package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import java.math.BigDecimal

@Embeddable
class PhotoBibEmbed(
    @Column(name = "bib_number", nullable = false, length = 20)
    var bibNumber: String,

    @Column(name = "ocr_confidence", nullable = false, precision = 5, scale = 4)
    var ocrConfidence: BigDecimal,
) {
    override fun equals(other: Any?): Boolean = this === other ||
        (other is PhotoBibEmbed && bibNumber == other.bibNumber)

    override fun hashCode(): Int = bibNumber.hashCode()
}
