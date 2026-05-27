package com.quickpitik.mobile

import android.app.Application
import coil.ImageLoader
import coil.ImageLoaderFactory
import com.quickpitik.mobile.data.remote.HostRewriteInterceptor
import okhttp3.OkHttpClient

// Coil singleton entry point. Installs an ImageLoader whose OkHttp client
// runs HostRewriteInterceptor — see that file for the localhost → 10.0.2.2
// rationale. Without it, every AsyncImage on the emulator fails.
class QuickPitikApp : Application(), ImageLoaderFactory {
    override fun newImageLoader(): ImageLoader {
        val okHttp = OkHttpClient.Builder()
            .addInterceptor(HostRewriteInterceptor)
            .build()
        return ImageLoader.Builder(this)
            .okHttpClient(okHttp)
            .build()
    }
}
