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
    @Query("SELECT * FROM upload_queue ORDER BY id ASC")
    fun getAllRecords(): Flow<List<UploadRecord>>

    // The Capture tab observes these three instead of the whole table: at a
    // 1,000-frame event the full-table Flow re-emitted every row on every
    // status flip (millions of row objects on the main dispatcher).
    @Query("SELECT uploadStatus AS status, COUNT(*) AS count FROM upload_queue GROUP BY uploadStatus")
    fun getStatusCounts(): Flow<List<StatusCount>>

    @Query("SELECT * FROM upload_queue ORDER BY id DESC LIMIT :limit")
    fun getRecentRecords(limit: Int): Flow<List<UploadRecord>>

    @Query("SELECT errorMessage FROM upload_queue WHERE uploadStatus = 'FAILED' ORDER BY id DESC LIMIT 1")
    fun getLatestFailedMessage(): Flow<String?>

    // Pruning the COMPLETED ledger: rows older than an event's upload window
    // can never dedupe a re-import (the backend would refuse the upload
    // anyway), so their thumbnails and rows are dead weight.
    @Query("SELECT filePath FROM upload_queue WHERE uploadStatus = 'COMPLETED' AND captureTimestamp < :before")
    suspend fun getCompletedPathsBefore(before: Long): List<String>

    @Query("DELETE FROM upload_queue WHERE uploadStatus = 'COMPLETED' AND captureTimestamp < :before")
    suspend fun deleteCompletedBefore(before: Long): Int

    // Newest first, so a just-shot live frame uploads ahead of an old
    // card-import backlog instead of waiting behind it.
    @Query("SELECT * FROM upload_queue WHERE uploadStatus = :status ORDER BY id DESC")
    suspend fun getRecordsWithStatus(status: String): List<UploadRecord>

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
            "errorMessage = NULL WHERE uploadStatus = 'FAILED'"
    )
    suspend fun requeueFailed(): Int

    /**
     * Records for [eventId] that should block a card photo from being re-imported:
     * anything still in flight (QUEUED / UPLOADING) or already landed (COMPLETED).
     * FAILED is intentionally excluded so the user can retry from the card after
     * a transient backend hiccup.
     */
    @Query(
        "SELECT * FROM upload_queue WHERE eventId = :eventId AND " +
            "uploadStatus IN ('QUEUED', 'UPLOADING', 'COMPLETED')"
    )
    suspend fun getActiveOrCompletedForEvent(eventId: String): List<UploadRecord>
}
