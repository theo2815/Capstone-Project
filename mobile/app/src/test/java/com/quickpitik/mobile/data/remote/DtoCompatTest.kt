package com.quickpitik.mobile.data.remote

import com.google.gson.Gson
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Plan 2 (race events) only adds JSON fields. Every build in the field parses
 * with plain Gson (RetrofitClient's GsonConverterFactory), which ignores unknown
 * fields and leaves a missing Kotlin field null. These pin both directions.
 */
class DtoCompatTest {
    private val gson = Gson()

    @Test fun `current DTOs ignore unknown keys and parse every new field`() {
        val event = gson.fromJson(EVENT_JSON, EventDto::class.java)
        assertEquals("cebu-ride", event.slug)
        assertEquals(4, event.photoCount)

        val summary = gson.fromJson(SUMMARY_JSON, PhotographerEventSummaryDto::class.java)
        assertEquals("change_pending", summary.reviewStatus)
        assertEquals("race_edit", summary.pendingChange?.get("kind"))

        val detail = gson.fromJson(DETAIL_JSON, PhotographerEventDetailDto::class.java)
        assertEquals(125.0, detail.pricePerPhoto, 0.0)

        val cart = gson.fromJson(CART_JSON, CartItemDto::class.java)
        assertEquals(150.0, cart.price, 0.0)
    }

    @Test fun `an old build's classes parse today's payload`() {
        val event = gson.fromJson(EVENT_JSON, OldEventDto::class.java)
        assertEquals("cebu-ride", event.slug)
        assertEquals(4, event.photoCount)

        val summary = gson.fromJson(SUMMARY_JSON, OldPhotographerEventSummaryDto::class.java)
        assertEquals("change_pending", summary.reviewStatus)
        assertEquals("race_edit", summary.pendingChange?.get("kind"))

        val detail = gson.fromJson(DETAIL_JSON, OldPhotographerEventDetailDto::class.java)
        assertEquals(125.0, detail.pricePerPhoto, 0.0)

        val cart = gson.fromJson(CART_JSON, OldCartItemDto::class.java)
        assertEquals(150.0, cart.price, 0.0)
    }

    @Test fun `new DTOs parse a pre-Plan-2 payload`() {
        val event = gson.fromJson(
            """{"id":"e1","slug":"s","name":"Run","date":"2026-01-10","location":"Cebu","bannerUrl":null,
               "photoCount":1,"participantCount":0,"status":"ACTIVE"}""",
            EventDto::class.java,
        )
        assertNull(event.pricePerPhoto)
        assertNull(event.priceVaries)
        assertNull(event.eventType)

        val detail = gson.fromJson(
            """{"id":"e1","slug":"s","name":"Run","date":"2026-01-10","location":"Cebu","pricePerPhoto":125.0}""",
            EventDetailDto::class.java,
        )
        assertNull(detail.priceVaries)
        assertNull(detail.eventType)

        val cart = gson.fromJson(
            """{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":125.0,"bib":null,"eventName":"Run",
               "eventSlug":"run","tone":null,"time":null}""",
            CartItemDto::class.java,
        )
        assertNull(cart.priceAtAdd)
        assertEquals(125.0, cart.price, 0.0)
    }

    // Today's server answers, with every new field and an unknown event type.
    private companion object {
        const val EVENT_JSON = """{"id":"e1","slug":"cebu-ride","name":"Cebu Ride","date":"2027-01-10","location":"Cebu",
               "bannerUrl":null,"photoCount":4,"participantCount":0,"status":"ACTIVE",
               "pricePerPhoto":125.0,"priceVaries":true,"eventType":"CYCLING","someFutureField":{"x":1}}"""
        const val RACE = """"submittedByMe":true,"canEditRace":false,"checkReasons":["NO_RACE_WORD"],"hidden":false,
            "eventType":"ROAD_RACE","racePrice":{"floorPhp":125.0,"ceilingPhp":250.0,"myPricePhp":150.0,"effectivePhp":150.0,"nextChangeAt":null},
            "pendingChange":{"kind":"race_edit","fields":{"title":"New name"},"requestedBy":"u1","requestedAt":"2027-01-01T00:00:00Z"}"""
        const val SUMMARY_JSON = """{"id":"e1","slug":"s","name":"Run","date":"2027-01-10","location":"Cebu","state":"open",
               "photoCount":1,"salesCount":0,"revenueKept":0.0,"reviewStatus":"change_pending",$RACE}"""
        const val DETAIL_JSON = """{"id":"e1","slug":"s","name":"Run","date":"2027-01-10","location":"Cebu","state":"open",
               "photoCount":1,"salesCount":0,"revenueKept":0.0,"pricePerPhoto":125.0,$RACE}"""
        const val CART_JSON = """{"photoId":"p1","eventId":"e1","thumbnailUrl":null,"price":150.0,"bib":null,"eventName":"Run",
               "eventSlug":"run","tone":null,"time":null,"available":true,"priceAtAdd":125.0}"""
    }
}
