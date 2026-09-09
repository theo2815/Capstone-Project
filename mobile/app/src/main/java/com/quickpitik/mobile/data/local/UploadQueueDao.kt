package com.quickpitik.mobile.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/** One row of `getStatusCounts()` — a status and how many rows carry it. */
data class StatusCount(val status: String, val count: Int)

@Dao
interface UploadQueueDao {
    @Query("UPDATE upload_queue SET uploadStatus = 'COMPLETED', errorMessage = NULL, uploadedPhotoId = :photoId, uploadedBackend = :backend, directPhotoId = NULL, directKey = NULL, directBackend = NULL WHERE id = :id")
    suspend fun completeUpload(id: Long, photoId: String, backend: String)

    @Query("SELECT * FROM upload_queue WHERE photographerId = :owner AND uploadStatus = 'COMPLETED' AND uploadedPhotoId IS NOT NULL AND uploadedBackend = :backend AND sourceKey IS NOT NULL AND contentHash IS NOT NULL AND localDeletedAt IS NULL ORDER BY id ASC")
    suspend fun cleanupCandidates(owner: String, backend: String): List<UploadRecord>

    @Query("UPDATE upload_queue SET localDeletedAt = :at WHERE id = :id AND uploadStatus = 'COMPLETED'")
    suspend fun markLocalDeleted(id: Long, at: Long)

    @Query("SELECT COUNT(*) FROM upload_queue WHERE filePath = :path AND id != :id AND localDeletedAt IS NULL")
    suspend fun otherFileReferences(path: String, id: Long): Int

    @Query("UPDATE upload_queue SET contentHash = :hash WHERE id = :id")
    suspend fun setContentHash(id: Long, hash: String)

    @Query("UPDATE upload_queue SET directPhotoId = :photoId, directKey = :key, directBackend = :backend WHERE id = :id")
    suspend fun checkpointDirectUpload(id: Long, photoId: String?, key: String?, backend: String?)

    @Query("UPDATE upload_queue SET retryAfter = :after WHERE id = :id")
    suspend fun defer(id: Long, after: Long)

    @Query("SELECT COALESCE(MAX(retryAfter), 0) FROM upload_queue WHERE photographerId = :owner AND uploadStatus = 'QUEUED'")
    suspend fun retryNotBefore(owner: String): Long
    @Query("SELECT * FROM upload_queue WHERE photographerId = :owner AND eventId = :event AND contentHash = :hash AND localDeletedAt IS NULL AND (uploadStatus IN ('QUEUED', 'UPLOADING') OR (uploadStatus = 'COMPLETED' AND uploadedPhotoId IS NOT NULL)) LIMIT 1")
    suspend fun findContent(owner: String, event: String, hash: String): UploadRecord?

    @Query("SELECT * FROM upload_queue WHERE photographerId = :owner AND uploadStatus = :status AND id > :afterId ORDER BY id ASC LIMIT 24")
    suspend fun getOwnedRecords(owner: String, status: String, afterId: Long): List<UploadRecord>

    @Query("UPDATE upload_queue SET uploadStatus = 'QUEUED' WHERE photographerId = :owner AND uploadStatus = 'UPLOADING'")
    suspend fun recoverOwned(owner: String)
    @Query("SELECT * FROM upload_queue WHERE (:owner IS NULL OR photographerId = :owner) ORDER BY id ASC")
    fun getAllRecords(owner: String? = null): Flow<List<UploadRecord>>

    // The Capture tab observes these three instead of the whole table: at a
    // 1,000-frame event the full-table Flow re-emitted every row on every
    // status flip (millions of row objects on the main dispatcher).
    @Query("SELECT uploadStatus AS status, COUNT(*) AS count FROM upload_queue WHERE (:owner IS NULL OR photographerId = :owner) GROUP BY uploadStatus")
    fun getStatusCounts(owner: String? = null): Flow<List<StatusCount>>

    @Query("SELECT * FROM upload_queue WHERE (:owner IS NULL OR photographerId = :owner) ORDER BY id DESC LIMIT :limit")
    fun getRecentRecords(limit: Int, owner: String? = null): Flow<List<UploadRecord>>

    @Query("SELECT errorMessage FROM upload_queue WHERE uploadStatus = 'FAILED' AND (:owner IS NULL OR photographerId = :owner) ORDER BY id DESC LIMIT 1")
    fun getLatestFailedMessage(owner: String? = null): Flow<String?>

    // Only retire rows whose local copy was explicitly removed after backup verification.
    @Query("DELETE FROM upload_queue WHERE uploadStatus = 'COMPLETED' AND localDeletedAt < :before AND (:owner IS NULL OR photographerId = :owner)")
    suspend fun deleteCompletedBefore(before: Long, owner: String? = null): Int

    // Newest first, so a just-shot live frame uploads ahead of an old
    // card-import backlog instead of waiting behind it.
    @Query("SELECT * FROM upload_queue WHERE uploadStatus = :status AND (:owner IS NULL OR photographerId = :owner) ORDER BY id DESC")
    suspend fun getRecordsWithStatus(status: String, owner: String? = null): List<UploadRecord>

    @Query("SELECT * FROM upload_queue WHERE id = :id LIMIT 1")
    suspend fun getRecordById(id: Long): UploadRecord?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertRecord(record: UploadRecord): Long

    @Update
    suspend fun updateRecord(record: UploadRecord)

    @Query("DELETE FROM upload_queue WHERE id = :id")
    suspend fun deleteRecordById(id: Long)

    @Query("UPDATE upload_queue SET uploadStatus = :status, errorMessage = :error WHERE id = :id")
    suspend fun updateStatus(id: Long, status: String, error: String?)

    @Query("UPDATE upload_queue SET retryCount = retryCount + 1 WHERE id = :id")
    suspend fun incrementRetryCount(id: Long)

    @Query("DELETE FROM upload_queue WHERE uploadStatus = :status")
    suspend fun deleteByStatus(status: String): Int

    /** Sends every FAILED row back through the queue with a fresh retry budget. */
    @Query(
        "UPDATE upload_queue SET uploadStatus = 'QUEUED', retryCount = 0, " +
            "errorMessage = NULL, retryAfter = 0 WHERE uploadStatus = 'FAILED' AND (:owner IS NULL OR photographerId = :owner)"
    )
    suspend fun requeueFailed(owner: String? = null): Int

    /**
     * Records for [eventId] that should block a card photo from being re-imported:
     * anything still in flight (QUEUED / UPLOADING) or already landed (COMPLETED).
     * FAILED is intentionally excluded so the user can retry from the card after
     * a transient backend hiccup.
     */
    @Query(
        "SELECT * FROM upload_queue WHERE eventId = :eventId AND (:owner IS NULL OR photographerId = :owner) AND " +
            "uploadStatus IN ('QUEUED', 'UPLOADING', 'COMPLETED')"
    )
    suspend fun getActiveOrCompletedForEvent(eventId: String, owner: String? = null): List<UploadRecord>
}
