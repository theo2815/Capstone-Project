package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Embeddable
import java.math.BigDecimal

/** `photo_bibs.bib_number` is VARCHAR(20); longer detected words are dropped, never cut. */
const val MAX_BIB_LENGTH = 20

@Embeddable
class PhotoBibEmbed(
    @Column(name = "bib_number", nullable = false, length = MAX_BIB_LENGTH)
    var bibNumber: String,

    @Column(name = "ocr_confidence", nullable = false, precision = 5, scale = 4)
    var ocrConfidence: BigDecimal,
) {
    override fun equals(other: Any?): Boolean = this === other ||
        (other is PhotoBibEmbed && bibNumber == other.bibNumber)

    override fun hashCode(): Int = bibNumber.hashCode()
}
