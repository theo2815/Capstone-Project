package com.quickpitik.mobile.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "upload_queue")
data class UploadRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val filePath: String,
    val eventId: String,
    val photographerId: String,
    val captureTimestamp: Long,
    val uploadStatus: String = "QUEUED", // QUEUED, UPLOADING, FAILED, COMPLETED
    val retryCount: Int = 0,
    val errorMessage: String? = null
)
