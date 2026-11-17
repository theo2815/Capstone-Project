package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.remote.PhotographerEventSummaryDto
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RaceFormTest {
    @Test fun `race type requires region and province`() {
        val input = OwnedEventFormInput(eventType = "ROAD_RACE")
        assertEquals(listOf("Pick a region.", "Pick a province."), raceFormErrors(input))
        assertEquals(listOf("Pick a province."), raceFormErrors(input.copy(regionCode = "07")))
        assertTrue(raceFormErrors(input.copy(regionCode = "07", provinceCode = "0722")).isEmpty())
    }

    @Test fun `Other needs no region`() {
        assertTrue(raceFormErrors(OwnedEventFormInput(eventType = "OTHER")).isEmpty())
        assertTrue(raceFormErrors(OwnedEventFormInput()).isEmpty())
    }

    private val race = PhotographerEventSummaryDto("race", "race", "Cebu Run", "2026-11-11", "Cebu", "live", 0, 0, 0.0,
        eventType = "ROAD_RACE", submittedByMe = true)

    @Test fun `submitted races show their review and hidden state without owner controls`() {
        assertEquals("Your race", submittedRaceLabel(race))
        assertEquals("Pending review", submittedRaceLabel(race.copy(reviewStatus = "pending")))
        assertEquals("Sent back", submittedRaceLabel(race.copy(reviewStatus = "rejected")))
        assertEquals("Change pending", submittedRaceLabel(race.copy(reviewStatus = "change_pending")))
        assertEquals("Hidden from /events", submittedRaceLabel(race.copy(hidden = true)))
        assertEquals(null, submittedRaceLabel(race.copy(submittedByMe = false)))
    }

    @Test fun `picker keeps coverage restrictions and the public price label`() {
        val public = race.copy(reviewStatus = "approved", hidden = null, pricePerPhoto = 50.0, priceVaries = true)
        val covered = race.copy(reviewStatus = "pending", hidden = true)
        val merged = uploadPickerEvents(listOf(public), listOf(covered)).single()
        assertEquals("pending", merged.reviewStatus)
        assertEquals(true, merged.hidden)
        assertEquals(50.0, merged.pricePerPhoto)
        assertEquals(true, merged.priceVaries)
        assertEquals("Hidden", raceUploadNotice(merged))
        assertEquals("Waiting for an admin", raceUploadNotice(covered.copy(hidden = false)))
    }
}
