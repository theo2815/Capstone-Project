package com.quickpitik.mobile.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(entities = [UploadRecord::class, CameraBaseline::class, CameraTransfer::class], version = 2, exportSchema = true)
abstract class AppDatabase : RoomDatabase() {
    abstract fun uploadQueueDao(): UploadQueueDao
    abstract fun cameraTransferDao(): CameraTransferDao

    companion object {
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE upload_queue ADD COLUMN contentHash TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE upload_queue ADD COLUMN sourceKey TEXT DEFAULT NULL")
                db.execSQL("ALTER TABLE upload_queue ADD COLUMN originalFilename TEXT DEFAULT NULL")
                db.execSQL("CREATE TABLE IF NOT EXISTS camera_baseline (scope TEXT NOT NULL PRIMARY KEY)")
                db.execSQL("CREATE TABLE IF NOT EXISTS camera_transfer (`key` TEXT NOT NULL PRIMARY KEY, scope TEXT NOT NULL, owner TEXT NOT NULL, eventId TEXT NOT NULL, sourceKey TEXT NOT NULL, handle INTEGER NOT NULL, filename TEXT NOT NULL, expectedBytes INTEGER, status TEXT NOT NULL, attempts INTEGER NOT NULL, retryAfter INTEGER NOT NULL, error TEXT, filePath TEXT, uploadId INTEGER)")
                db.execSQL("CREATE INDEX IF NOT EXISTS index_camera_transfer_scope_status ON camera_transfer (scope, status)")
            }
        }
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "quickpitik_db"
                ).addMigrations(MIGRATION_1_2).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
