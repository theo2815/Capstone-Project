package com.quickpitik.mobile.data.repository

import com.quickpitik.mobile.data.local.UploadQueueDao
import com.quickpitik.mobile.data.local.UploadRecord
import kotlinx.coroutines.flow.Flow

class PhotoRepositoryImpl(
    private val uploadQueueDao: UploadQueueDao
) : PhotoRepository {

    override fun getQueuedPhotos(): Flow<List<UploadRecord>> {
        return uploadQueueDao.getAllRecords()
    }

    override suspend fun enqueuePhoto(record: UploadRecord) {
        uploadQueueDao.insertRecord(record)
    }

    override suspend fun updateStatus(id: Long, status: String, errorMessage: String?) {
        uploadQueueDao.updateStatus(id, status, errorMessage)
    }

    override suspend fun deleteRecord(id: Long) {
        uploadQueueDao.deleteRecordById(id)
    }
}
