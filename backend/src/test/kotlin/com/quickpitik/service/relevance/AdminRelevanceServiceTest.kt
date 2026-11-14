package com.quickpitik.service.relevance

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.RelevanceProperties
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.RelevanceCase
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.RelevanceAlertRepository
import com.quickpitik.repository.RelevanceAlertRow
import com.quickpitik.repository.RelevanceCaseRepository
import com.quickpitik.repository.RelevanceCaseRow
import com.quickpitik.repository.SamplePhotoRow
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.Pageable
import java.time.Clock
import java.time.Duration
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class AdminRelevanceServiceTest {

    private val cases = Mockito.mock(RelevanceCaseRepository::class.java)
    private val alerts = Mockito.mock(RelevanceAlertRepository::class.java)
    private val batches = Mockito.mock(BatchRelevanceRepository::class.java)
    private val storage = Mockito.mock(StorageService::class.java)
    private val now: Instant = Instant.parse("2003-04-05T06:07:08.123456Z")
    private val props = RelevanceProperties(junk = RelevanceProperties.Junk(blurCutoff = 40.0, darkCutoff = 5.0))
    private val service = AdminRelevanceService(cases, alerts, batches, props, storage, StorageProperties(),
        Clock.fixed(now, ZoneOffset.UTC))

    @Test
    fun `the hidden tab lists cases with an action, marked would-have in record-only`() {
        val hidden = row(action = "HIDE_ALL", appliedMode = "RECORD_ONLY", measured = 300, junk = 240)
        stubTab("hidden", hidden)

        val page = service.listCases("hidden", PaginationParams.of(0, 50))

        assertEquals(listOf(hidden.caseId), page.items.map { it.caseId })
        assertTrue(page.items.single().wouldHave)
        assertEquals(80, page.items.single().junkPercent)
        assertFalse(row(action = "NONE", appliedMode = "RECORD_ONLY").let { stubTab("all", it); service.listCases("all", PaginationParams.of(0, 50)) }
            .items.single().wouldHave, "no action, nothing would have happened")
    }

    @Test
    fun `the can't tell tab lists only open can't tell cases`() {
        stubTab("cant_tell", row(verdict = "CANT_TELL", action = "NONE", appliedMode = "RECORD_ONLY"))

        val page = service.listCases("cant_tell", PaginationParams.of(0, 50))

        assertEquals(listOf("CANT_TELL"), page.items.map { it.verdict })
        Mockito.verify(cases).pageCurrentCases(Mockito.eq("cant_tell") ?: "", anyPageable())
        assertEquals(0, service.listCases("appeals", PaginationParams.of(0, 50)).items.size, "appeals come with part 2")
    }

    @Test
    fun `a case shows at most twelve junk and twelve clean samples and never an original`() {
        val r = row(action = "HIDE_JUNK", appliedMode = "RECORD_ONLY")
        Mockito.`when`(cases.caseRow(r.caseId)).thenReturn(r)
        Mockito.`when`(cases.findById(r.caseId)).thenReturn(Optional.of(RelevanceCase(id = r.caseId, batchId = r.batchId,
            trigger = "QUIET", uploadAt = OffsetDateTime.now(), batchPhotos = 600, stage = "APPLIED", evidenceUsed = "junk_share,copy_rule")))
        // 20 blurry photos and 20 sharp ones; half of each have only a thumbnail.
        val sample = (0 until 40).map { i -> photo(sharpness = if (i < 20) 10.0 else 90.0, watermarked = i % 2 == 0) }
        Mockito.`when`(cases.measuredSample(r.caseId)).thenReturn(sample)
        Mockito.`when`(storage.presignedGetUrl(Mockito.anyString(), anyDuration())).thenAnswer { "https://signed/${it.getArgument<String>(0)}" }

        val detail = service.case(r.caseId)

        assertEquals(12, detail.junkSamples.size)
        assertEquals(12, detail.cleanSamples.size)
        assertTrue(detail.junkSamples.all { it.blurry == true })
        assertTrue(detail.cleanSamples.all { it.blurry == false })
        val urls = (detail.junkSamples + detail.cleanSamples).map { it.thumbnailUrl!! }
        assertTrue(urls.all { it.contains("/watermark/") || it.contains("/thumb/") })
        assertFalse(urls.any { it.contains("/original/") })
        assertEquals(listOf("junk_share", "copy_rule"), detail.evidenceUsed)
    }

    @Test
    fun `marking an alert seen stamps the admin and the time once`() {
        val alertId = UUID.randomUUID()
        val admin = UUID.randomUUID()
        Mockito.`when`(alerts.alertRow(alertId)).thenReturn(alertRow(alertId, seenAt = now))

        val dto = service.markSeen(admin, alertId)

        Mockito.verify(alerts, Mockito.times(1)).markSeen(alertId, admin, OffsetDateTime.ofInstant(now, ZoneOffset.UTC))
        assertEquals(now.toString(), dto.seenAt)
    }

    private fun stubTab(tab: String, vararg rows: RelevanceCaseRow) {
        Mockito.`when`(cases.pageCurrentCases(Mockito.eq(tab) ?: tab, anyPageable())).thenReturn(PageImpl(rows.toList()))
    }

    private fun row(
        verdict: String = "JUNK_DUMP",
        action: String? = null,
        appliedMode: String? = null,
        measured: Int? = null,
        junk: Int? = null,
    ): RelevanceCaseRow {
        val id = UUID.randomUUID()
        val v = verdict; val a = action; val m = appliedMode; val ms = measured; val j = junk
        return object : RelevanceCaseRow {
            override val caseId = id
            override val batchId = UUID.randomUUID()
            override val eventId = UUID.randomUUID()
            override val eventName = "Rel Run"
            override val photographerHandle = "rel"
            override val photographerName = "Rel"
            override val verdict = v
            override val verdictNote: String? = null
            override val action = a
            override val appliedMode = m
            override val batchPhotos = 600
            override val measured = ms
            override val junk = j
            override val decidedAt = now
        }
    }

    private fun photo(sharpness: Double, watermarked: Boolean): SamplePhotoRow {
        val id = UUID.randomUUID()
        return object : SamplePhotoRow {
            override val photoId = id
            override val watermarkKey = if (watermarked) "events/e/photos/$id/watermark/w.jpg" else null
            override val thumbnailKey = "events/e/photos/$id/thumb/t.jpg"
            override val fgSharpness = sharpness
            override val fgBrightness = 50.0
            override val sharpness = sharpness
            override val brightness = 50.0
            override val personConf = 0.99
            override val raceConf = 0.9
        }
    }

    private fun alertRow(id: UUID, seenAt: Instant?): RelevanceAlertRow {
        val seen = seenAt
        return object : RelevanceAlertRow {
            override val alertId = id
            override val caseId: UUID? = UUID.randomUUID()
            override val eventId = UUID.randomUUID()
            override val eventName = "Rel Run"
            override val photographerHandle = "rel"
            override val photographerName = "Rel"
            override val kind = "COPIED"
            override val wouldHave = true
            override val photos: Int? = 40
            override val createdAt: Instant? = now
            override val seenAt = seen
        }
    }

    private fun anyPageable(): Pageable = Mockito.any(Pageable::class.java) ?: Pageable.unpaged()
    private fun anyDuration(): Duration = Mockito.any(Duration::class.java) ?: Duration.ZERO
}
