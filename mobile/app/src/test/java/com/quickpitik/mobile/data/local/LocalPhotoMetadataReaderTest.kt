package com.quickpitik.mobile.data.local

import android.app.Application
import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.google.gson.Gson
import com.google.gson.JsonParser
import com.quickpitik.mobile.data.remote.PhotoMetadataFieldsDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.support.MainDispatcherRule
import com.quickpitik.mobile.support.MockBackend
import com.quickpitik.mobile.support.awaitFirst
import com.quickpitik.mobile.support.ok
import com.quickpitik.mobile.ui.common.PhotoMetadataState
import com.quickpitik.mobile.ui.common.PhotoMetadataViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.RecordedRequest
import okio.Buffer
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.RuleChain
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(RobolectricTestRunner::class)
class LocalPhotoMetadataReaderTest {
    private val context = ApplicationProvider.getApplicationContext<Context>()
    private val session = SessionManager.getInstance(context)
    private val main = MainDispatcherRule()
    private val backend = MockBackend()
    @get:Rule val rules: RuleChain = RuleChain.outerRule(main).around(backend)

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
        val outside = File(context.filesDir, "camera_outside.jpg").apply {
            LocalPhotoMetadataReaderTest::class.java.getResourceAsStream("/photo-metadata/complete-exif.jpg")!!.use { input ->
                outputStream().use(input::copyTo)
            }
        }
        assertTrue(outside.length() > 4)
        assertEquals("EOS R6", outside.inputStream().use(reader::extract).cameraModel)
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

    @Test fun `verified cleanup racing spool open yields local fields or matching receipt fallback`() = runBlocking {
        session.saveSession("token", "PHOTOGRAPHER", "Name", "owner")
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        try {
            val file = record("complete-exif.jpg")
            val bytes = File(file.filePath).readBytes()
            val hash = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            val uploaded = file.copy(uploadStatus = "COMPLETED", contentHash = hash, sourceKey = "source",
                uploadedPhotoId = "cloud-photo", uploadedBackend = RetrofitClient.BASE_URL)
            val row = uploaded.copy(id = db.uploadQueueDao().insertRecord(uploaded))
            val cloudVerified = CountDownLatch(1)
            backend.server.dispatcher = object : Dispatcher() {
                override fun dispatch(request: RecordedRequest): MockResponse = when {
                    request.path?.endsWith("/download") == true ->
                        ok("""{"url":"${backend.server.url("/original")}"}""")
                    request.path == "/original" -> {
                        cloudVerified.countDown()
                        MockResponse().setBody(Buffer().write(bytes))
                    }
                    request.path?.endsWith("/cloud-photo/metadata") == true ->
                        ok("""{"photoId":"cloud-photo","status":"ready","originalFilename":"race-original.jpg","fileSizeBytes":${bytes.size},"fields":{"cameraMake":"Canon","cameraModel":"EOS R6","lens":null,"aperture":null,"exposureTime":null,"iso":null,"author":null,"copyright":null,"takenAtLocal":null,"takenAtOffset":null,"flash":null,"exposureMode":null,"whiteBalance":null}}""")
                    else -> MockResponse().setResponseCode(404)
                }
            }
            val readStarted = CountDownLatch(1)
            val reader = LocalPhotoMetadataReader(context)
            val vm = PhotoMetadataViewModel(context as Application,
                getRecord = { db.uploadQueueDao().getRecordById(it) },
                readLocal = { readStarted.countDown(); reader.read(it) })
            val cleanup = TetherStorageCleanup(context, db)
            UploadSpool.filesMutex.lock()
            var gateHeld = true
            val removal = try {
                val deleting = async(Dispatchers.IO) { cleanup.remove(row) }
                assertTrue(cloudVerified.await(5, TimeUnit.SECONDS))
                vm.selectLocal(row.id)
                assertTrue(readStarted.await(5, TimeUnit.SECONDS))
                UploadSpool.filesMutex.unlock()
                gateHeld = false
                withTimeout(5_000) { deleting.await() }
            } finally {
                if (gateHeld) UploadSpool.filesMutex.unlock()
            }
            assertEquals(bytes.size.toLong(), removal.bytes)
            assertTrue(removal.recorded)
            assertFalse(File(row.filePath).exists())
            assertNotNull(db.uploadQueueDao().getRecordById(row.id)?.localDeletedAt)
            val state = try {
                vm.state.awaitFirst {
                    it is PhotoMetadataState.LocalData || it is PhotoMetadataState.Data
                }
            } catch (e: Exception) {
                throw AssertionError("state=${vm.state.value}; requests=${backend.server.requestCount}", e)
            }
            when (state) {
                is PhotoMetadataState.LocalData -> {
                    assertEquals("EOS R6", state.metadata.fields?.cameraModel)
                    assertEquals(bytes.size.toLong(), state.metadata.fileSizeBytes)
                }
                is PhotoMetadataState.Data -> {
                    assertEquals("cloud-photo", state.metadata.photoId)
                    assertEquals("EOS R6", state.metadata.fields?.cameraModel)
                }
                else -> fail("Cleanup race lost metadata")
            }
        } finally {
            db.close()
        }
    }
}
