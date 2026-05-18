package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.local.UploadRecord
import kotlinx.coroutines.flow.Flow

interface PhotoRepository {
    fun getQueuedPhotos(): Flow<List<UploadRecord>>
    suspend fun enqueuePhoto(record: UploadRecord)
    suspend fun updateStatus(id: Long, status: String, errorMessage: String? = null)
    suspend fun deleteRecord(id: Long)
}
