package com.quickpitik.service.photos

import com.quickpitik.entity.Photo

// The filename a runner's browser saves a clean original as. One home for
// the order-download and free-download paths (V46).
object PhotoFilenames {
    private val UNSAFE = Regex("[^A-Za-z0-9._-]")

    fun downloadFilenameOf(photo: Photo): String {
        val bib = photo.bibs.minByOrNull { it.bibNumber }?.bibNumber
        val tag = if (!bib.isNullOrBlank()) "bib-$bib" else "untagged-${photo.id.toString().take(8)}"
        return "quickpitik-$tag.jpg".replace(UNSAFE, "_")
    }
}
