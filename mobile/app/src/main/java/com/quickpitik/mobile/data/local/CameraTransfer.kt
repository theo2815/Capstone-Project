package com.quickpitik.mobile.data.local

import androidx.room.*

@Entity(tableName = "camera_baseline")
data class CameraBaseline(@PrimaryKey val scope: String)

data class CameraRecoverySummary(val pending: Int = 0, val failed: Int = 0, val error: String? = null)

@Entity(tableName = "camera_transfer", indices = [Index(value = ["scope", "status"])])
data class CameraTransfer(
    @PrimaryKey val key: String,
    val scope: String,
    val owner: String,
    val eventId: String,
    val sourceKey: String,
    val handle: Long,
    val filename: String,
    val expectedBytes: Long?,
    val status: String = "PENDING",
    val attempts: Int = 0,
    val retryAfter: Long = 0,
    val error: String? = null,
    val filePath: String? = null,
    val uploadId: Long? = null,
)

@Dao
interface CameraTransferDao {
    @Query("SELECT COUNT(*) AS pending, COALESCE(SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END), 0) AS failed, (SELECT error FROM camera_transfer WHERE owner = :owner AND eventId = :event AND status = 'FAILED' ORDER BY retryAfter DESC LIMIT 1) AS error FROM camera_transfer WHERE owner = :owner AND eventId = :event AND status IN ('PENDING', 'FAILED', 'TRANSFERRING')")
    fun observeRecovery(owner: String, event: String): kotlinx.coroutines.flow.Flow<CameraRecoverySummary>
    @Query("UPDATE camera_transfer SET retryAfter = 0, status = 'PENDING', error = NULL WHERE owner = :owner AND eventId = :event AND status = 'FAILED'")
    suspend fun retry(owner: String, event: String)
    @Query("SELECT COUNT(*) FROM camera_baseline WHERE scope = :scope")
    suspend fun baselineExists(scope: String): Int
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertBaseline(baseline: CameraBaseline)
    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insert(transfers: List<CameraTransfer>)
    @Update
    suspend fun update(transfer: CameraTransfer)
    @Query("SELECT * FROM camera_transfer WHERE `key` = :key")
    suspend fun get(key: String): CameraTransfer?
    @Query("SELECT * FROM camera_transfer WHERE scope = :scope")
    suspend fun forScope(scope: String): List<CameraTransfer>
    @Query("SELECT * FROM camera_transfer WHERE owner = :owner AND status = 'TRANSFERRING'")
    suspend fun interrupted(owner: String): List<CameraTransfer>
}
