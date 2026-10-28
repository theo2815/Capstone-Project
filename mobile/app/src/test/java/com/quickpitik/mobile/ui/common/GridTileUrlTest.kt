package com.quickpitik.mobile.ui.common

import com.google.gson.Gson
import com.quickpitik.mobile.data.remote.PhotoDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

// FS-12: photo grids load the backend's small grid tile; photos without one
// (published before it existed, or an older backend) keep the preview.
class GridTileUrlTest {
    private val gson = Gson()
    private val base = """{"id":"p1","bib":"101","km":null,"time":"07:30","tone":0,"span":"default","price":150.0,
        "imageUrl":"https://cdn.test/watermark.jpg","alt":null"""

    @Test fun `the grid tile is used when the backend sends one`() {
        val photo = gson.fromJson("""$base,"gridUrl":"https://cdn.test/grid.jpg"}""", PhotoDto::class.java)

        assertEquals("https://cdn.test/grid.jpg", photo.gridTileUrl())
    }

    @Test fun `older responses without gridUrl fall back to the preview`() {
        val photo = gson.fromJson("$base}", PhotoDto::class.java)

        assertNull(photo.gridUrl)
        assertEquals("https://cdn.test/watermark.jpg", photo.gridTileUrl())
    }
}
