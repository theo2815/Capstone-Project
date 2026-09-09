package com.quickpitik.mobile.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import androidx.room.ColumnInfo

@Entity(tableName = "upload_queue")
data class UploadRecord(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val filePath: String,
    val eventId: String,
    val photographerId: String,
    val captureTimestamp: Long,
    val uploadStatus: String = "QUEUED", // QUEUED, UPLOADING, FAILED, COMPLETED
    val retryCount: Int = 0,
    val errorMessage: String? = null,
    @ColumnInfo(defaultValue = "NULL") val contentHash: String? = null,
    @ColumnInfo(defaultValue = "NULL") val sourceKey: String? = null,
    @ColumnInfo(defaultValue = "NULL") val originalFilename: String? = null,
    @ColumnInfo(defaultValue = "NULL") val directPhotoId: String? = null,
    @ColumnInfo(defaultValue = "NULL") val directKey: String? = null,
    @ColumnInfo(defaultValue = "NULL") val directBackend: String? = null,
    @ColumnInfo(defaultValue = "0") val retryAfter: Long = 0,
)
