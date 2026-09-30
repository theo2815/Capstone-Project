package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.data.remote.PhotoMetadataFieldsDto
import com.google.gson.Gson
import com.google.gson.JsonParser
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File

@RunWith(RobolectricTestRunner::class)
class LocalPhotoMetadataReaderTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val session = SessionManager.getInstance(context)

    private fun record(name: String, owner: String = "owner"): UploadRecord {
        val file = File(UploadSpool.dir(context), "camera_${System.nanoTime()}.jpg")
        javaClass.getResourceAsStream("/photo-metadata/$name")!!.use { input -> file.outputStream().use(input::copyTo) }
        return UploadRecord(filePath = file.absolutePath, eventId = "event", photographerId = owner,
            captureTimestamp = 1, originalFilename = "folder\\race-original.jpg")
    }

    @Test fun `owned retained original yields M2 fields and original file facts offline`() = runBlocking {
        session.saveSession("token", "PHOTOGRAPHER", "Name", "owner")
        val row = record("complete-exif.jpg")
        val result = LocalPhotoMetadataReader(context).read(row)
        assertEquals("ready", result.status)
        assertEquals("Canon", result.fields?.cameraMake)
        assertEquals("EOS R6", result.fields?.cameraModel)
        assertEquals("RF 70-200mm F2.8", result.fields?.lens)
        assertEquals("1/500 s", result.fields?.exposureTime)
        assertEquals(400L, result.fields?.iso)
        assertEquals("Did not fire", result.fields?.flash)
        assertEquals("Auto", result.fields?.exposureMode)
        assertEquals("Auto", result.fields?.whiteBalance)
        assertEquals("race-original.jpg", result.originalFilename)
        assertEquals(File(row.filePath).length(), result.fileSizeBytes)
    }

    @Test fun `editorial tags override camera text and preserve local capture offset`() = runBlocking {
        session.saveSession("token", "PHOTOGRAPHER", "Name", "owner")
        val fields = LocalPhotoMetadataReader(context).read(record("edited-iptc-xmp.jpg")).fields
        assertEquals("Élodie Cebu", fields?.author)
        assertEquals("© Élodie 2026", fields?.copyright)
        assertEquals("2026-09-28T06:35:20.125", fields?.takenAtLocal)
        assertEquals("+08:00", fields?.takenAtOffset)
        assertEquals("Manual", fields?.exposureMode)
    }

    @Test fun `outside spool and foreign account cannot open original`() = runBlocking {
        session.saveSession("token", "PHOTOGRAPHER", "Name", "owner")
        val reader = LocalPhotoMetadataReader(context)
        assertEquals("unavailable", reader.read(record("complete-exif.jpg", "another")).status)
        val outside = File(context.filesDir, "camera_outside.jpg").apply { writeBytes(byteArrayOf(1)) }
        assertEquals("unavailable", reader.read(record("complete-exif.jpg").copy(filePath = outside.absolutePath)).status)
    }

    @Test fun `all M2 EXIF IPTC and embedded XMP fixtures match expected fields`() {
        val gson = Gson()
        val expected = javaClass.getResourceAsStream("/photo-metadata/expected.json")!!.reader().use {
            JsonParser.parseReader(it).asJsonObject.getAsJsonObject("vectors")
        }
        val reader = LocalPhotoMetadataReader(context)
        expected.entrySet().forEach { (name, json) ->
            val actual = javaClass.getResourceAsStream("/photo-metadata/$name")!!.use(reader::extract)
            val wanted = gson.fromJson(json, PhotoMetadataFieldsDto::class.java)
            assertEquals(name, wanted, actual)
        }
        assertTrue(runCatching {
            javaClass.getResourceAsStream("/photo-metadata/malformed.jpg")!!.use(reader::extract)
        }.isFailure)
    }

    @Test fun `large embedded XMP is bounded and private tags stay excluded`() {
        val result = javaClass.getResourceAsStream("/photo-metadata/xmp-payload-limit.jpg")!!.use {
            LocalPhotoMetadataReader(context).extract(it)
        }
        val serialized = Gson().toJson(result)
        assertTrue(serialized.toByteArray().size <= 8192)
        for (privateTag in listOf("GPS_SENTINEL", "SERIAL_SENTINEL", "PRIVATE_PATH_SENTINEL")) {
            assertFalse(serialized.contains(privateTag))
        }
    }
}
