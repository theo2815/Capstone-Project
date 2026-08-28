package com.quickpitik.service.photographer

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.PhotographerMessage
import com.quickpitik.entity.RunnerMessage
import com.quickpitik.repository.PhotographerMessageRepository
import com.quickpitik.repository.RunnerMessageRepository
import com.quickpitik.service.runner.RunnerMessageReaderService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.Pageable
import java.util.UUID
import kotlin.test.assertEquals

// Both inboxes used to return every non-removed row a user had ever received.
// They now page. The wire shape stays a bare array — only the row count is
// bounded — so no website or mobile change was required.
class MessagePaginationTest {

    private val photographerRepo: PhotographerMessageRepository =
        Mockito.mock(PhotographerMessageRepository::class.java)
    private val runnerRepo: RunnerMessageRepository =
        Mockito.mock(RunnerMessageRepository::class.java)

    private fun <T> anyArg(): T = Mockito.any()

    @Test
    fun `photographer inbox forwards offset and limit to the repository`() {
        val userId = UUID.randomUUID()
        var seenId: UUID? = null
        var seenPage: Pageable? = null
        Mockito.`when`(
            photographerRepo.findByPhotographerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(
                anyArg(),
                anyArg(),
            ),
        ).thenAnswer { inv ->
            seenId = inv.getArgument(0)
            seenPage = inv.getArgument(1)
            emptyList<PhotographerMessage>()
        }

        PhotographerMessageService(photographerRepo).list(userId, PaginationParams.of(40, 25))

        assertEquals(userId, seenId)
        assertEquals(40L, seenPage?.offset)
        assertEquals(25, seenPage?.pageSize)
    }

    @Test
    fun `runner inbox forwards offset and limit to the repository`() {
        val userId = UUID.randomUUID()
        var seenId: UUID? = null
        var seenPage: Pageable? = null
        Mockito.`when`(
            runnerRepo.findByRunnerIdAndRemovedAtIsNullOrderByCreatedAtDescIdAsc(anyArg(), anyArg()),
        ).thenAnswer { inv ->
            seenId = inv.getArgument(0)
            seenPage = inv.getArgument(1)
            emptyList<RunnerMessage>()
        }

        RunnerMessageReaderService(runnerRepo).list(userId, PaginationParams.of(0, 100))

        assertEquals(userId, seenId)
        assertEquals(0L, seenPage?.offset)
        assertEquals(100, seenPage?.pageSize)
    }

    @Test
    fun `the 100-row controller default survives PaginationParams clamping`() {
        // MAX_LIMIT is 200, so the inbox default passes through unchanged. If the
        // cap were ever lowered below 100 this test flags the silent truncation.
        assertEquals(100, PaginationParams.of(null, 100).limit)
    }

    // The controllers set X-Total-Count from these counts so the web inbox knows
    // the true total behind its capped page (the body stays a bare array).
    @Test
    fun `photographer inbox count returns the un-removed repository total`() {
        val userId = UUID.randomUUID()
        Mockito.`when`(photographerRepo.countByPhotographerIdAndRemovedAtIsNull(userId))
            .thenReturn(137L)

        assertEquals(137L, PhotographerMessageService(photographerRepo).count(userId))
    }

    @Test
    fun `runner inbox count returns the un-removed repository total`() {
        val userId = UUID.randomUUID()
        Mockito.`when`(runnerRepo.countByRunnerIdAndRemovedAtIsNull(userId))
            .thenReturn(42L)

        assertEquals(42L, RunnerMessageReaderService(runnerRepo).count(userId))
    }
}
