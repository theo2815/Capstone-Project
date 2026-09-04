package com.quickpitik.mobile.data.remote

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class EventGallerySnapshotContractTest {
    private lateinit var server: MockWebServer
    private lateinit var api: QuickPitikApi

    @Before
    fun setUp() {
        server = MockWebServer().apply { start() }
        api = Retrofit.Builder()
            .baseUrl(server.url("/"))
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(QuickPitikApi::class.java)
    }

    @After
    fun tearDown() = server.shutdown()

    @Test
    fun `event gallery carries the snapshot into later pages`() = runBlocking {
        val snapshot = "2026-09-03T12:34:56Z"
        server.enqueue(
            MockResponse().setBody(
                """{"success":true,"data":{"items":[],"total":0,"offset":60,"limit":60,"snapshotAt":"$snapshot"}}""",
            ),
        )

        val response = api.getEventPhotos(
            slug = "cebu-marathon",
            offset = 60,
            limit = 60,
            snapshotAt = snapshot,
        )

        assertEquals(snapshot, response.data?.snapshotAt)
        assertEquals(snapshot, server.takeRequest().requestUrl?.queryParameter("snapshotAt"))
    }
}
