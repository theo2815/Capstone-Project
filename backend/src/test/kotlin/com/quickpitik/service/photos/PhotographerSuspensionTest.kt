package com.quickpitik.service.photos

import com.quickpitik.entity.Photo
import com.quickpitik.repository.UserRepository
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.math.BigDecimal
import java.util.UUID
import kotlin.test.assertEquals

class PhotographerSuspensionTest {
    private val users: UserRepository = Mockito.mock(UserRepository::class.java)
    private val suspension = PhotographerSuspension(users)

    @Test
    fun `suspendedOwnerIds returns only suspended owners and runs no query for an empty input`() {
        val active = UUID.randomUUID()
        val suspended = UUID.randomUUID()
        Mockito.`when`(users.findSuspendedIdsIn(setOf(active, suspended))).thenReturn(listOf(suspended))

        val photos = listOf(photo(active), photo(suspended), photo(suspended), photo(null))
        assertEquals(setOf(suspended), suspension.suspendedOwnerIds(photos))

        assertEquals(emptySet<UUID>(), suspension.suspendedOwnerIds(emptyList()))
        assertEquals(emptySet<UUID>(), suspension.suspendedOwnerIds(listOf(photo(null))))
        Mockito.verify(users, Mockito.times(1)).findSuspendedIdsIn(Mockito.anyCollection())
    }

    private fun photo(owner: UUID?) = Photo(
        eventId = UUID.randomUUID(),
        photographerId = owner,
        s3Key = "t/${UUID.randomUUID()}.jpg",
        pricePhp = BigDecimal("125.00"),
    )
}
