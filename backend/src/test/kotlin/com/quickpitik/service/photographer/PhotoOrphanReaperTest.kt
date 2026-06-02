package com.quickpitik.service.photographer

import com.quickpitik.config.AiApiProperties
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.service.ai.AiApiClient
import com.quickpitik.service.ai.AiPersonRef
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.time.OffsetDateTime
import java.util.UUID

// Orphan reaper: deletes only the ai-api persons that are BOTH unreferenced by a
// live photo AND older than the grace window, and never lets one failure stop
// the sweep. The age gate is what keeps an in-flight enroll from being reaped.
class PhotoOrphanReaperTest {

    private lateinit var photoRepo: PhotoRepository
    private lateinit var aiApiClient: AiApiClient

    // Mockito.any() returns null at runtime; declared as a (platform) T so Kotlin
    // inserts no null-assertion — usable as a matcher for a non-null param.
    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        photoRepo = Mockito.mock(PhotoRepository::class.java)
        aiApiClient = Mockito.mock(AiApiClient::class.java)
    }

    private fun reaper(props: AiApiProperties) = PhotoOrphanReaper(photoRepo, aiApiClient, props)

    @Test
    fun `reap when ai-api disabled does nothing`() {
        reaper(AiApiProperties(enabled = false)).reap()

        Mockito.verifyNoInteractions(photoRepo)
        Mockito.verifyNoInteractions(aiApiClient)
    }

    @Test
    fun `reap when reaper disabled does nothing`() {
        reaper(AiApiProperties(enabled = true, reaperEnabled = false)).reap()

        Mockito.verifyNoInteractions(photoRepo)
        Mockito.verifyNoInteractions(aiApiClient)
    }

    @Test
    fun `reap with no indexed events never lists persons`() {
        Mockito.`when`(photoRepo.findEventsWithFacePersons(anyArg())).thenReturn(emptyList())

        reaper(AiApiProperties(enabled = true)).reap()

        Mockito.verifyNoInteractions(aiApiClient)
    }

    @Test
    fun `reap deletes only unreferenced persons older than the grace window`() {
        val eventId = UUID.randomUUID()
        val old = OffsetDateTime.now().minusHours(1)   // before the 30-min cutoff
        val fresh = OffsetDateTime.now()               // inside the grace window
        val referenced = "person-referenced"
        val orphan = "person-orphan"
        val justEnrolled = "person-fresh"

        Mockito.`when`(photoRepo.findEventsWithFacePersons(anyArg())).thenReturn(listOf(eventId))
        Mockito.`when`(aiApiClient.listPersonsForEvent(eventId)).thenReturn(
            listOf(
                AiPersonRef(referenced, old),
                AiPersonRef(orphan, old),
                AiPersonRef(justEnrolled, fresh),
            ),
        )
        Mockito.`when`(photoRepo.findReferencedAiPersonIds(eventId)).thenReturn(listOf(referenced))

        reaper(AiApiProperties(enabled = true)).reap()

        // The orphan is gone…
        Mockito.verify(aiApiClient).deleteFacesPerson(orphan)
        // …but a still-referenced person is left alone…
        Mockito.verify(aiApiClient, Mockito.never()).deleteFacesPerson(referenced)
        // …and so is one enrolled inside the grace window (could be mid-index).
        Mockito.verify(aiApiClient, Mockito.never()).deleteFacesPerson(justEnrolled)
    }

    @Test
    fun `a failed delete does not abort the remaining orphans`() {
        val eventId = UUID.randomUUID()
        val old = OffsetDateTime.now().minusHours(1)
        val first = "orphan-1"
        val second = "orphan-2"

        Mockito.`when`(photoRepo.findEventsWithFacePersons(anyArg())).thenReturn(listOf(eventId))
        Mockito.`when`(aiApiClient.listPersonsForEvent(eventId)).thenReturn(
            listOf(AiPersonRef(first, old), AiPersonRef(second, old)),
        )
        Mockito.`when`(photoRepo.findReferencedAiPersonIds(eventId)).thenReturn(emptyList())
        Mockito.doThrow(RuntimeException("ai-api down")).`when`(aiApiClient).deleteFacesPerson(first)

        reaper(AiApiProperties(enabled = true)).reap()

        // First throws, but the second is still attempted.
        Mockito.verify(aiApiClient).deleteFacesPerson(first)
        Mockito.verify(aiApiClient).deleteFacesPerson(second)
    }
}
