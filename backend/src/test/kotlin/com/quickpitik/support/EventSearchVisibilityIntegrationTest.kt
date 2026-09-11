package com.quickpitik.support

import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.EventVisibility
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.data.domain.PageRequest
import java.time.LocalDate
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * V46 event isolation. `EventRepository.search` is the single query behind BOTH
 * the public `/events` list and every photographer's upload picker, so its
 * `createdBy IS NULL AND visibility = PUBLIC` clause is the enforcement that
 * keeps photographer-owned events off the public list and out of a foreign
 * photographer's picker. That clause lives only in SQL — the Mockito unit suite
 * mocks the repository and can never exercise it — so this real-DB test is the
 * one place the guarantee is actually pinned.
 */
class EventSearchVisibilityIntegrationTest : PostgresIntegrationTest() {

    @Autowired
    private lateinit var eventRepository: EventRepository

    @Autowired
    private lateinit var userRepository: UserRepository

    @Test
    fun `public search returns admin PUBLIC events and never photographer-owned or unlisted events`() {
        // A unique marker in every name so search() isolates to just this test's
        // rows — the Postgres container is shared across classes, so never assume
        // an empty table (see PostgresIntegrationTest docblock).
        val marker = "vis-${UUID.randomUUID()}"
        val owner = userRepository.save(
            User(
                email = "owner-${UUID.randomUUID()}@test.local",
                passwordHash = "stub",
                name = "Owner",
                role = Role.PHOTOGRAPHER,
            ),
        ).id

        val adminPublic = save(marker, createdBy = null, visibility = EventVisibility.PUBLIC)
        val adminUnlisted = save(marker, createdBy = null, visibility = EventVisibility.UNLISTED)
        val ownedPublic = save(marker, createdBy = owner, visibility = EventVisibility.PUBLIC)
        val ownedUnlisted = save(marker, createdBy = owner, visibility = EventVisibility.UNLISTED)

        val hits = eventRepository.search(
            statuses = listOf(EventStatus.ACTIVE, EventStatus.COMPLETED, EventStatus.ARCHIVED),
            search = marker,
            city = "",
            dateFrom = LocalDate.of(1900, 1, 1),
            dateTo = LocalDate.of(9999, 12, 31),
            pageable = PageRequest.of(0, 100),
        ).content.map { it.id }.toSet()

        // The only event that belongs on the public list / upload picker.
        assertTrue(adminPublic in hits, "admin PUBLIC event must appear on /events")
        // The V46 fix: photographer-owned events (createdBy != null) never leak
        // onto the public list — public OR unlisted, same query drives both surfaces.
        assertFalse(ownedPublic in hits, "photographer-owned PUBLIC event must be excluded")
        assertFalse(ownedUnlisted in hits, "photographer-owned UNLISTED event must be excluded")
        // Pre-existing visibility filter still holds for admin events too.
        assertFalse(adminUnlisted in hits, "admin UNLISTED event must be excluded")
    }

    // The other SQL-only V46 rule: a DRAFT — a pending or rejected photographer
    // submission — has no public page even by direct link, while an UNLISTED
    // live event does; that is what "link-only" means (2026-09-11 audit).
    @Test
    fun `findPublicBySlug hides a DRAFT submission and resolves an UNLISTED live event`() {
        val marker = "slug-${UUID.randomUUID()}"
        val owner = userRepository.save(
            User(
                email = "owner-${UUID.randomUUID()}@test.local",
                passwordHash = "stub",
                name = "Owner",
                role = Role.PHOTOGRAPHER,
            ),
        ).id
        val draft = save(marker, createdBy = owner, visibility = EventVisibility.PUBLIC, status = EventStatus.DRAFT)
        val unlisted = save(marker, createdBy = owner, visibility = EventVisibility.UNLISTED)

        assertNull(eventRepository.findPublicBySlug(slugOf(draft)), "a DRAFT submission must not resolve publicly")
        assertEquals(unlisted, eventRepository.findPublicBySlug(slugOf(unlisted))?.id)
    }

    private fun slugOf(id: UUID): String = eventRepository.findById(id).orElseThrow().slug

    private fun save(
        marker: String,
        createdBy: UUID?,
        visibility: EventVisibility,
        status: EventStatus = EventStatus.ACTIVE,
    ): UUID =
        eventRepository.save(
            Event(
                slug = "$marker-${UUID.randomUUID()}",
                name = "$marker race",
                date = LocalDate.of(2026, 6, 15),
                location = "Cebu City",
                status = status,
                createdBy = createdBy,
                visibility = visibility,
            ),
        ).id
}
