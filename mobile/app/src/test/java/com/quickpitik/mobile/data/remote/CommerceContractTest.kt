package com.quickpitik.mobile.data.remote

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

// Phase 0 contract test: the coupon / free-event / owned-event fields added to the
// DTOs must map to the EXACT backend JSON keys — a wrong name deserializes to a
// silent null and the coupon/discount vanishes (a money bug). These feed
// representative backend payloads through the real Gson converter and assert the
// new fields land. Mirrors EventGallerySnapshotContractTest.
class CommerceContractTest {
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
    fun `PhotoDto parses coupon and free-event fields`() = runBlocking {
        server.enqueue(
            MockResponse().setBody(
                """{"success":true,"data":{"items":[
                    {"id":"p1","bib":"1023","km":10,"time":"08:30","tone":0,"span":"portrait",
                     "price":125.0,"imageUrl":"http://img","alt":null,
                     "couponCode":"SAVE20","couponPercentOff":20,"couponPrice":100.0,
                     "free":false,"downloadUrl":null},
                    {"id":"p2","bib":null,"km":null,"time":"09:00","tone":0,"span":"landscape",
                     "price":0.0,"imageUrl":"http://img2","alt":null,
                     "free":true,"downloadUrl":"http://clean/p2"}
                ],"total":2,"offset":0,"limit":100}}""",
            ),
        )

        val photos = api.getEventPhotos(slug = "cebu-marathon").data!!.items
        val paid = photos[0]
        assertEquals("SAVE20", paid.couponCode)
        assertEquals(20, paid.couponPercentOff)
        assertEquals(100.0, paid.couponPrice!!, 0.0001)
        assertFalse(paid.free)
        assertNull(paid.downloadUrl)

        val free = photos[1]
        assertTrue(free.free)
        assertEquals("http://clean/p2", free.downloadUrl)
    }

    @Test
    fun `CouponPreviewDto parses per-item discount and totals`() = runBlocking {
        server.enqueue(
            MockResponse().setBody(
                """{"success":true,"data":{
                    "code":"SAVE20","percentOff":20,
                    "photographerName":"Juan","photographerHandle":"juan",
                    "items":[{"photoId":"p1","price":125.0,"discount":25.0,"couponCode":"SAVE20","percentOff":20}],
                    "eligibleCount":1,"discountTotal":25.0}}""",
            ),
        )

        val preview = api.previewCoupon(
            request = CouponPreviewRequest(photoIds = listOf("p1"), code = "SAVE20"),
        ).data!!
        assertEquals("SAVE20", preview.code)
        assertEquals(1, preview.eligibleCount)
        assertEquals(25.0, preview.discountTotal, 0.0001)
        assertEquals(25.0, preview.items.first().discount, 0.0001)
        assertEquals(125.0, preview.items.first().price, 0.0001)
    }

    @Test
    fun `OrderResponse parses the zero-peso inline-settlement shape`() = runBlocking {
        // ₱0 checkout: FULFILLED with a returnToken and no redirectUrl.
        server.enqueue(
            MockResponse().setBody(
                """{"success":true,"data":{
                    "id":"o1","status":"FULFILLED",
                    "items":[{"photoId":"p1","price":0.0,"downloadUrl":"http://clean/p1"}],
                    "totalAmount":0.0,"checkoutTotalAmount":0.0,"paymentMethod":"gcash","createdAt":"2026-09-06T00:00:00Z",
                    "returnToken":"tok123"}}""",
            ),
        )

        val order = api.createOrder(
            token = "Bearer x",
            idempotencyKey = "k1",
            request = CreateOrderRequest(
                items = listOf(CreateOrderItem("p1", "e1")),
                paymentMethod = "gcash",
                recipientEmail = null,
            ),
        ).data!!
        assertEquals("FULFILLED", order.status)
        assertEquals(0.0, order.totalAmount, 0.0001)
        assertEquals("tok123", order.returnToken)
        assertNull(order.redirectUrl)
    }

    @Test
    fun `PhotographerEventSummaryDto parses the V46 owned-event fields`() = runBlocking {
        server.enqueue(
            MockResponse().setBody(
                """{"success":true,"data":{"items":[
                    {"id":"e1","slug":"cebu","name":"Cebu Fun Run","date":"2026-09-01","location":"Cebu",
                     "bannerUrl":null,"state":"live","photoCount":10,"salesCount":2,"revenueKept":300.0,
                     "ownedByMe":true,"visibility":"unlisted","pricingMode":"free","watermarkPolicy":"own",
                     "reviewStatus":"pending","reviewNote":"needs a cover",
                     "pendingChange":{"pricePerPhoto":150}}
                ],"total":1,"offset":0,"limit":200}}""",
            ),
        )

        val event = api.getPhotographerEvents(token = "Bearer x").data!!.items.first()
        assertTrue(event.ownedByMe)
        assertEquals("unlisted", event.visibility)
        assertEquals("free", event.pricingMode)
        assertEquals("pending", event.reviewStatus)
        assertEquals("needs a cover", event.reviewNote)
        assertNotNull(event.pendingChange)
        assertEquals(150.0, event.pendingChange!!["pricePerPhoto"])
    }
}
