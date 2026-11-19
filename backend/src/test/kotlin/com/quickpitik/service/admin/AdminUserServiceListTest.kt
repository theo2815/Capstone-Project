package com.quickpitik.service.admin

import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.repository.*
import com.quickpitik.service.auth.RefreshTokenService
import com.quickpitik.service.photographer.PhotographerSettingsService
import com.quickpitik.service.profile.UserDtoMapper
import com.quickpitik.service.reference.RegionsService
import com.quickpitik.service.runner.RunnerMessagesService
import com.quickpitik.service.storage.StorageService
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.Pageable
import kotlin.test.assertEquals

// AX-43: the status tab is filtered inside the query, so the page and its total agree.
class AdminUserServiceListTest {
    private val queries = mutableListOf<List<Any?>>()
    private val userRepository = Mockito.mock(UserRepository::class.java) { call ->
        if (call.method.name == "searchForAdmin") {
            queries += call.arguments.toList()
            PageImpl(emptyList<User>(), call.arguments.last() as Pageable, 7)
        } else null
    }

    private fun service() = AdminUserService(userRepository, Mockito.mock(PhotographerSettingsRepository::class.java),
        Mockito.mock(SocialLinkRepository::class.java), Mockito.mock(PayoutAccountRepository::class.java),
        Mockito.mock(AdminDecisionLogRepository::class.java), Mockito.mock(RegionsService::class.java),
        Mockito.mock(AdminDecisionLogService::class.java), Mockito.mock(StorageService::class.java), StorageProperties(),
        Mockito.mock(UserDtoMapper::class.java), Mockito.mock(RunnerMessagesService::class.java),
        Mockito.mock(RefreshTokenService::class.java), Mockito.mock(AdminActionCommandService::class.java),
        Mockito.mock(PhotographerMessageRepository::class.java), Mockito.mock(RunnerMessageRepository::class.java),
        Mockito.mock(PhotographerSettingsService::class.java))

    @Test
    fun `a status tab goes to the query and the total is the query's own count`() {
        val page = service().list("photographer", "Pending", "  Ana ", PaginationParams.of(0, 50))

        assertEquals(listOf<Any?>(Role.PHOTOGRAPHER, "ana", "pending"), queries.single().take(3))
        assertEquals(7, page.total)
    }

    @Test
    fun `no status or an unknown one filters nothing, as before`() {
        service().list(null, null, null, PaginationParams.of(0, 50))
        service().list(null, "everything", null, PaginationParams.of(0, 50))

        assertEquals(listOf<Any?>("", ""), queries.map { it[2] })
    }
}
