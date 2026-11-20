package com.quickpitik.support

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.PhotographerSettings
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.VerificationStatus
import com.quickpitik.repository.PhotographerSettingsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminUserService
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals

// AX-43: the users list filters status in the query (pages and total agree) and
// searches the photographer's handle and brand name as well as name and email.
class AdminUsersListIntegrationTest : PostgresIntegrationTest() {
    @Autowired private lateinit var service: AdminUserService
    @Autowired private lateinit var users: UserRepository
    @Autowired private lateinit var settings: PhotographerSettingsRepository

    @Test
    fun `a status tab pages the filtered set and counts only that status`() {
        val tag = UUID.randomUUID().toString().take(8)
        val pending = (1..3).map { photographer(tag, VerificationStatus.PENDING) }
        repeat(2) { photographer(tag, VerificationStatus.APPROVED) }
        val suspended = photographer(tag, VerificationStatus.APPROVED, suspended = true)
        val noSettings = photographer(tag, status = null)
        val rejected = photographer(tag, VerificationStatus.REJECTED)
        val runner = users.saveAndFlush(User(email = "runner-$tag@test.local", passwordHash = "unused", name = "Runner $tag", role = Role.RUNNER))

        val first = service.list("PHOTOGRAPHER", "pending", tag, PaginationParams.of(0, 2))
        val second = service.list("PHOTOGRAPHER", "pending", tag, PaginationParams.of(2, 2))
        assertEquals(3, first.total)
        assertEquals(2, first.items.size)
        assertEquals(1, second.items.size)
        assertEquals(pending.map { it.id }.toSet(), (first.items + second.items).map { it.userId }.toSet())

        fun ids(status: String, role: String? = null) = service.list(role, status, tag, PaginationParams.of(0, 50)).let {
            assertEquals(it.items.size.toLong(), it.total)
            it.items.map { row -> row.userId }.toSet()
        }
        // Runners count as approved; a suspended photographer leaves the approved tab.
        assertEquals(2, ids("approved", "PHOTOGRAPHER").size)
        assertEquals(true, runner.id in ids("approved"))
        assertEquals(setOf(suspended.id), ids("suspended"))
        assertEquals(setOf(noSettings.id), ids("incomplete"))
        assertEquals(setOf(rejected.id), ids("rejected"))
        assertEquals(9, service.list(null, null, tag, PaginationParams.of(0, 50)).total)
    }

    // AX-46: suspended wins over every other status, so a suspended photographer sits in one tab only.
    @Test
    fun `a suspended photographer is only in the suspended tab`() {
        val tag = UUID.randomUUID().toString().take(8)
        val held = listOf(
            photographer(tag, VerificationStatus.PENDING, suspended = true),
            photographer(tag, status = null, suspended = true),
            photographer(tag, VerificationStatus.REJECTED, suspended = true),
        )

        fun ids(status: String) = service.list("PHOTOGRAPHER", status, tag, PaginationParams.of(0, 50)).let {
            assertEquals(it.items.size.toLong(), it.total)
            it.items.map { row -> row.userId }.toSet()
        }
        assertEquals(held.map { it.id }.toSet(), ids("suspended"))
        assertEquals(emptySet(), ids("pending"))
        assertEquals(emptySet(), ids("incomplete"))
        assertEquals(emptySet(), ids("rejected"))
    }

    @Test
    fun `search finds a photographer by handle and by brand name`() {
        val tag = UUID.randomUUID().toString().take(8)
        val photographer = photographer("someone", VerificationStatus.APPROVED, handle = "h$tag", brand = "Brand $tag Studio")

        assertEquals(listOf(photographer.id), service.list(null, null, "H$tag", PaginationParams.of(0, 50)).items.map { it.userId })
        assertEquals(listOf(photographer.id), service.list(null, null, "brand $tag", PaginationParams.of(0, 50)).items.map { it.userId })
    }

    // AX-44: % and _ in the search match only themselves.
    @Test
    fun `a percent or underscore in the search is plain text`() {
        val tag = UUID.randomUUID().toString().take(8)
        val offer = photographer("$tag 10% off", VerificationStatus.APPROVED)
        photographer("$tag 100", VerificationStatus.APPROVED)

        fun found(q: String) = service.list(null, null, q, PaginationParams.of(0, 50)).items.map { it.userId }.toSet()
        assertEquals(setOf(offer.id), found("$tag 10%"))
        assertEquals(emptySet(), found("$tag 10_"))
        assertEquals(2, found(tag).size)
    }

    private fun photographer(
        tag: String,
        status: VerificationStatus?,
        suspended: Boolean = false,
        handle: String? = null,
        brand: String? = null,
    ): User {
        val user = users.saveAndFlush(
            User(
                email = "users-list-${UUID.randomUUID()}@test.local",
                passwordHash = "unused",
                name = "Users List $tag",
                role = Role.PHOTOGRAPHER,
            ).also { if (suspended) it.suspendedAt = OffsetDateTime.now() },
        )
        if (status != null) {
            settings.saveAndFlush(
                PhotographerSettings(userId = user.id, verificationStatus = status).also {
                    it.handle = handle
                    it.brandName = brand
                },
            )
        }
        return user
    }
}
