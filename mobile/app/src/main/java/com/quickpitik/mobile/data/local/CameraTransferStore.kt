package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.room.withTransaction
import com.quickpitik.mobile.data.usb.ptp.PtpObjectInfo
import com.quickpitik.mobile.data.usb.ptp.fingerprint
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.io.IOException
import java.io.OutputStream
import java.security.DigestOutputStream
import java.security.MessageDigest
import java.util.UUID

class LowCameraStorage : IOException("Phone storage is nearly full. Free space to resume; pending photos remain on the camera.")

/** Durable boundary between camera bytes and the existing upload queue. */
class CameraTransferStore(private val context: Context, private val db: AppDatabase) {
    val dao get() = db.cameraTransferDao()

    fun transfer(owner: String, event: String, camera: String, handle: Long, info: PtpObjectInfo): CameraTransfer {
        val scope = fingerprint(owner, event, camera)
        val source = info.sourceKey(camera, handle)
        return CameraTransfer(fingerprint(scope, source), scope, owner, event, source, handle,
            info.filename, info.expectedSize, if (info.isJpeg && !info.isAssociation) "PENDING" else "EXCLUDED")
    }

    suspend fun reconcile(scope: String, objects: List<CameraTransfer>): List<CameraTransfer> = db.withTransaction {
        val first = dao.baselineExists(scope) == 0
        dao.insert(if (first) objects.map { it.copy(status = "BASELINE") } else objects)
        if (first) dao.insertBaseline(CameraBaseline(scope))
        dao.forScope(scope)
    }

    suspend fun fail(record: CameraTransfer, error: String) {
        val current = dao.get(record.key) ?: record
        dao.update(current.copy(status = "FAILED", attempts = current.attempts + 1,
            retryAfter = System.currentTimeMillis() + minOf(60_000L, 3000L * (current.attempts + 1)), error = error))
    }

    /** Persist the entire selection before USB I/O so an interrupted batch can be retried. */
    suspend fun selectForImport(records: List<CameraTransfer>) = persistence.withLock {
        db.withTransaction {
            dao.insert(records)
            for (record in records) {
                val current = dao.get(record.key) ?: continue
                if (current.status != "QUEUED")
                    dao.update(current.copy(status = "PENDING", retryAfter = 0, error = null))
            }
        }
    }

    /** Sony reuses one RAM handle. Each attempt is distinct until content hashes prove equality. */
    suspend fun saveTransient(
        owner: String, event: String, camera: String, handle: Long, info: PtpObjectInfo,
        write: (OutputStream) -> Unit,
    ): Long {
        if (!UploadSpool.hasRoom(context, info.expectedSize ?: 0)) throw LowCameraStorage()
        val source = "transient:${UUID.randomUUID()}"
        val record = transfer(owner, event, camera, handle, info).let {
            it.copy(key = fingerprint(it.scope, source), sourceKey = source)
        }
        try {
            return save(record, write)
        } catch (e: CancellationException) { throw e }
        catch (e: Exception) {
            val current = dao.get(record.key)
            // A complete renamed file remains recoverable; never overwrite its recovery state.
            if (current != null && !(current.status == "TRANSFERRING" && current.filePath?.let(::File)?.exists() == true))
                fail(current, "Temporary camera transfer interrupted. Import the backup from the camera card; this buffer handle cannot be retried safely.")
            throw e
        }
    }

    suspend fun save(record: CameraTransfer, write: (OutputStream) -> Unit): Long = persistence.withLock {
        require(record.owner.isNotBlank() && record.owner != "unknown") { "Sign in before importing photos." }
        dao.insert(listOf(record))
        val existing = dao.get(record.key)?.uploadId?.let { db.uploadQueueDao().getRecordById(it) }
        if (existing != null && existing.uploadStatus != "FAILED" && File(existing.filePath).exists()) return@withLock existing.id
        dao.update(record.copy(status = "PENDING", error = null))
        if (record.expectedBytes != null && record.expectedBytes > MAX_PUBLISH_BYTES) {
            fail(record, "This JPEG exceeds the 25 MiB upload limit. Keep the original on the card and select smaller JPEGs for subsequent captures.")
            throw IOException("Camera JPEG exceeds the upload size limit.")
        }
        val file = File(UploadSpool.dir(context), "camera_${record.key}.jpg")
        val partial = File(file.path + ".part")
        val working = record.copy(status = "TRANSFERRING", filePath = file.absolutePath, error = null)
        if (!UploadSpool.hasRoom(context, record.expectedBytes ?: 0)) throw LowCameraStorage()
        dao.update(working)
        try {
            val digest = MessageDigest.getInstance("SHA-256")
            val coroutine = currentCoroutineContext()
            partial.outputStream().use { raw ->
                val output = DigestOutputStream(raw.buffered(), digest)
                val bounded = object : OutputStream() {
                    var count = 0L
                    override fun write(b: Int) = write(byteArrayOf(b.toByte()), 0, 1)
                    override fun write(b: ByteArray, off: Int, len: Int) {
                        coroutine.ensureActive()
                        count += len
                        if (count > MAX_PUBLISH_BYTES) throw IOException("JPEG exceeds the 25 MiB upload limit. Choose a smaller camera JPEG setting.")
                        output.write(b, off, len)
                    }
                }
                write(bounded)
                output.flush()
                raw.fd.sync()
            }
            validate(partial, record.expectedBytes)
            if (!partial.renameTo(file)) throw IOException("Couldn't finalize the downloaded photo.")
            // A crash after rename leaves TRANSFERRING + the final file for recover().
            return@withLock enqueue(working, file, digest.digest().hex())
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            if (!file.exists()) fail(record, e.message ?: "Camera transfer failed.")
            throw e
        } finally { partial.delete() }
    }

    private suspend fun enqueue(record: CameraTransfer, file: File, hash: String): Long {
        val result = db.withTransaction {
            val uploads = db.uploadQueueDao()
            val existing = uploads.findContent(record.owner, record.eventId, hash)
                ?.takeIf { File(it.filePath).exists() }
            val id = existing?.id ?: uploads.insertRecord(UploadRecord(
                filePath = file.absolutePath, eventId = record.eventId, photographerId = record.owner,
                captureTimestamp = System.currentTimeMillis(), contentHash = hash,
                sourceKey = record.sourceKey, originalFilename = record.filename,
            ))
            val retainedPath = existing?.filePath ?: file.absolutePath
            dao.update(record.copy(status = "QUEUED", uploadId = id, error = null, filePath = retainedPath))
            id to retainedPath
        }
        if (result.second != file.absolutePath) file.delete()
        return result.first
    }

    suspend fun recover(owner: String) = persistence.withLock {
        for (record in dao.interrupted(owner)) {
            val file = record.filePath?.let(::File) ?: continue
            File(file.path + ".part").delete()
            if (file.exists()) {
                try {
                    validate(file, record.expectedBytes)
                    val digest = MessageDigest.getInstance("SHA-256")
                    file.inputStream().use { input ->
                        val bytes = ByteArray(16384)
                        while (true) { val n = input.read(bytes); if (n < 0) break; digest.update(bytes, 0, n) }
                    }
                    enqueue(record, file, digest.digest().hex())
                } catch (e: CancellationException) { throw e }
                catch (e: Exception) { fail(record, e.message ?: "Interrupted photo needs a retry.") }
            } else fail(record, if (record.sourceKey.startsWith("transient:"))
                "Temporary camera transfer interrupted. Import the backup from the camera card; this buffer handle cannot be retried safely."
                else "Transfer interrupted. Reconnect the camera to retry.")
        }
    }

    companion object {
        // ponytail: one camera writer per app; use per-transfer locks if multi-camera ingestion is added.
        private val persistence = Mutex()
        // Existing direct-upload contract; RAW publishing needs a backend contract change.
        const val MAX_PUBLISH_BYTES = 25L * 1024 * 1024
        internal fun validate(file: File, expected: Long?) {
            val size = file.length()
            if (size < 4 || size > MAX_PUBLISH_BYTES || (expected != null && size != expected))
                throw IOException("Incomplete or oversized camera JPEG ($size bytes).")
            file.inputStream().use { if (it.read() != 0xff || it.read() != 0xd8) throw IOException("Camera file is not a JPEG. Select JPEG or RAW+JPEG.") }
        }
        private fun ByteArray.hex() = joinToString("") { "%02x".format(it) }
    }
}
