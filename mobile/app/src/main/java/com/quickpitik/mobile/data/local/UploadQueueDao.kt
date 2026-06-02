package com.quickpitik.mobile.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface UploadQueueDao {
    @Query("SELECT * FROM upload_queue ORDER BY id ASC")
    fun getAllRecords(): Flow<List<UploadRecord>>

    @Query("SELECT * FROM upload_queue WHERE uploadStatus = :status ORDER BY id ASC")
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
