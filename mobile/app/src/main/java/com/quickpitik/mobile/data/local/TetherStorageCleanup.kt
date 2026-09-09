package com.quickpitik.mobile.data.local

import android.content.Context
import com.quickpitik.mobile.data.remote.RetrofitClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import okhttp3.Call
import okhttp3.Callback
import okhttp3.Request
import okhttp3.Response
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.security.MessageDigest

/** Only app-owned tether originals with a persisted backend receipt can enter cleanup. */
class TetherStorageCleanup(private val context: Context, private val db: AppDatabase) {
    data class Review(val records: List<UploadRecord>, val bytes: Long)
    data class Removal(val bytes: Long, val recorded: Boolean)
    private val session = SessionManager.getInstance(context)
    private val dao = db.uploadQueueDao()

    suspend fun review(): Review = withContext(Dispatchers.IO) {
        val owner = session.getUserEmail() ?: throw IOException("Sign in to review phone copies.")
        val records = dao.cleanupCandidates(owner, RetrofitClient.BASE_URL).filter { record ->
            runCatching { eligible(record) }.getOrDefault(false)
        }
        Review(records, records.sumOf { File(it.filePath).length() })
    }

    /** Call only for the confirmed snapshot. Failed verification always keeps the original. */
    suspend fun remove(record: UploadRecord): Removal = withContext(Dispatchers.IO) {
        requireActiveOwner(record)
        check(dao.getRecordById(record.id) == record && eligible(record)) { "Phone copy changed. Review it again." }
        val file = File(record.filePath)
        val coroutine = currentCoroutineContext()
        val local = file.inputStream().use { digest(it, CameraTransferStore.MAX_PUBLISH_BYTES) { coroutine.ensureActive() } }
        check(local.hash == record.contentHash) { "Phone original no longer matches its upload." }

        // Existing authenticated endpoint resolves the ORIGINAL, never a thumbnail or marketplace preview.
        val link = RetrofitClient.apiService.getPhotographerPhotoDownload(requireActiveOwner(record), record.uploadedPhotoId!!)
        check(link.success) { "Cloud original could not be confirmed." }
        val url = RetrofitClient.resolveImageUrl(link.data?.url)?.takeIf { it.isNotBlank() }
            ?: throw IOException("Cloud original is unavailable.")
        verifyRemote(url, local)

        UploadSpool.filesMutex.withLock {
            coroutine.ensureActive()
            requireActiveOwner(record)
            check(dao.getRecordById(record.id) == record && eligible(record)) { "Phone copy changed during verification." }
            check(dao.otherFileReferences(record.filePath, record.id) == 0) { "Another transfer still uses this phone copy." }
            val current = file.inputStream().use { digest(it, local.bytes) { coroutine.ensureActive() } }
            check(current == local) { "Phone original changed during verification." }
            coroutine.ensureActive()
            requireActiveOwner(record)
            // Cancellation cannot split these operations. A process crash between them leaves a
            // COMPLETED row with an absent file, which is excluded from later cleanup, never re-uploaded.
            withContext(NonCancellable) {
                check(file.delete()) { "Android could not remove this phone copy." }
                // The filesystem and Room cannot commit atomically. Report actual removal even
                // if a full/broken DB cannot record it; never tell the photographer the file stayed.
                Removal(local.bytes, runCatching { dao.markLocalDeleted(record.id, System.currentTimeMillis()) }.isSuccess)
            }
        }
    }

    private fun eligible(record: UploadRecord): Boolean {
        val file = File(record.filePath)
        return record.uploadStatus == "COMPLETED" && record.localDeletedAt == null &&
            !record.uploadedPhotoId.isNullOrBlank() && record.uploadedBackend == RetrofitClient.BASE_URL &&
            !record.sourceKey.isNullOrBlank() && !record.eventId.isBlank() &&
            record.contentHash?.matches(Regex("[0-9a-f]{64}")) == true &&
            UploadSpool.owns(context, file) && file.isFile && file.length() in 4..CameraTransferStore.MAX_PUBLISH_BYTES
    }

    private fun requireActiveOwner(record: UploadRecord): String {
        val token = session.getAccessToken()
        check(token != null && session.authSnapshot(token)?.owner == record.photographerId &&
            RetrofitClient.BASE_URL == record.uploadedBackend) { "Account or server changed. Phone copies kept." }
        return "Bearer $token"
    }

    private data class Content(val bytes: Long, val hash: String)

    private fun digest(input: InputStream, maxBytes: Long, checkActive: () -> Unit): Content {
        val hash = MessageDigest.getInstance("SHA-256")
        val buffer = ByteArray(64 * 1024)
        var bytes = 0L
        while (true) {
            checkActive()
            val count = input.read(buffer)
            if (count < 0) break
            bytes += count
            check(bytes <= maxBytes) { "Original size does not match." }
            hash.update(buffer, 0, count)
        }
        return Content(bytes, hash.digest().joinToString("") { "%02x".format(it) })
    }

    // ponytail: cleanup reads one original over the network per file. A server-verified checksum
    // receipt can replace this download if field measurements justify a backend contract change.
    private suspend fun verifyRemote(url: String, expected: Content) = suspendCancellableCoroutine<Unit> { continuation ->
        val request = Request.Builder().url(url).header("Accept-Encoding", "identity")
            .header("Cache-Control", "no-cache").build()
        val call = RetrofitClient.rawClient.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { continuation.resumeWith(Result.failure(e)) }
            override fun onResponse(call: Call, response: Response) {
                val result = runCatching {
                    response.use {
                        check(it.code == 200) { "Cloud original is unavailable." }
                        val body = it.body ?: throw IOException("Cloud original is empty.")
                        check(body.contentLength() == -1L || body.contentLength() == expected.bytes) { "Cloud original size differs." }
                        val remote = digest(body.byteStream(), expected.bytes) {
                            if (!continuation.isActive) throw IOException("Verification cancelled.")
                        }
                        check(remote == expected) { "Cloud original bytes differ. Phone copy kept." }
                    }
                }
                continuation.resumeWith(result)
            }
        })
    }
}
