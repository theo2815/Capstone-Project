package com.quickpitik.mobile.data.download

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import com.quickpitik.mobile.data.remote.HostRewriteInterceptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

// Streams a photo from a backend presigned URL into the device's MediaStore so
// it shows up in Google Photos / Gallery / Files — the mobile parity for the
// website's `<a download>` flow. minSdk 29 means we don't need
// WRITE_EXTERNAL_STORAGE (scoped storage via MediaStore handles it).
//
// IS_PENDING gates the file as "still being written" so other apps don't pick
// up a half-flushed JPEG; the flag is cleared in a second update once the body
// finishes streaming. RELATIVE_PATH lands the photo under
// /Pictures/QuickPitik/ so users find their purchases in one place.
object PhotoDownloader {

    private val http = OkHttpClient.Builder()
        .addInterceptor(HostRewriteInterceptor)
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .build()

    sealed class Result {
        data class Saved(val uri: Uri, val displayName: String) : Result()
        data class Failed(val message: String) : Result()
    }

    suspend fun saveToGallery(
        context: Context,
        url: String,
        filename: String,
    ): Result = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/QuickPitik")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: return@withContext Result.Failed("Couldn't open a slot in your gallery.")

        try {
            val request = Request.Builder().url(url).build()
            http.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    resolver.delete(uri, null, null)
                    return@withContext Result.Failed("Server returned ${response.code} for the photo.")
                }
                val body = response.body
                    ?: run {
                        resolver.delete(uri, null, null)
                        return@withContext Result.Failed("Server returned an empty response.")
                    }
                resolver.openOutputStream(uri).use { out ->
                    if (out == null) {
                        resolver.delete(uri, null, null)
                        return@withContext Result.Failed("Couldn't write the photo to your gallery.")
                    }
                    body.byteStream().use { input -> input.copyTo(out) }
                }
            }
        } catch (e: IOException) {
            resolver.delete(uri, null, null)
            return@withContext Result.Failed(e.localizedMessage ?: "Network error while downloading.")
        } catch (e: Exception) {
            resolver.delete(uri, null, null)
            return@withContext Result.Failed(e.localizedMessage ?: "Couldn't save the photo.")
        }

        val finalize = ContentValues().apply {
            put(MediaStore.Images.Media.IS_PENDING, 0)
        }
        resolver.update(uri, finalize, null, null)
        Result.Saved(uri, filename)
    }

    // Standard filename mirroring the website's buildPhotoDownloadFilename.
    // bib slug stays human-readable; untagged photos use the first 8 chars of
    // the photo UUID so duplicate runner-untagged shots don't collide.
    fun buildFilename(photoId: String, bib: String?): String {
        val tag = if (!bib.isNullOrBlank()) {
            "bib-${bib.lowercase()}"
        } else {
            "untagged-${photoId.take(8)}"
        }
        return "quickpitik-$tag.jpg"
    }
}
