package com.quickpitik.service.events

import com.quickpitik.common.PaginatedResponse
import com.quickpitik.config.AiApiProperties
import com.quickpitik.dto.photos.PhotoDto
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventPhotoAlert
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import com.quickpitik.entity.UserSelfie
import com.quickpitik.repository.EventPhotoAlertRepository
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.repository.UserSelfieRepository
import com.quickpitik.service.email.EmailService
import com.quickpitik.service.photos.PhotoSearchService
import com.quickpitik.service.storage.StorageService
import com.quickpitik.support.testTransactionTemplate
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.annotation.AnnotationTransactionAttributeSource
import org.springframework.transaction.support.AbstractPlatformTransactionManager
import org.springframework.transaction.support.DefaultTransactionStatus
import org.springframework.transaction.support.TransactionTemplate
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.Optional
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

// Mirrors OrderReceiptEmailClaimTest: the single-send guarantee for the
// "photos ready" mail. What matters is exactly one send per match, that a
// not-ready alert never spends its one claim, and that a failed send hands the
// claim back so the next sweep retries.
class EventPhotosReadyNotifierTest {

    private lateinit var alertRepository: EventPhotoAlertRepository
    private lateinit var eventRepository: EventRepository
    private lateinit var userRepository: UserRepository
    private lateinit var userSelfieRepository: UserSelfieRepository
    private lateinit var storageService: StorageService
    private lateinit var photoSearchService: PhotoSearchService
    private lateinit var emailService: EmailService

    @BeforeEach
    fun setUp() {
        alertRepository = Mockito.mock(EventPhotoAlertRepository::class.java)
        eventRepository = Mockito.mock(EventRepository::class.java)
        userRepository = Mockito.mock(UserRepository::class.java)
        userSelfieRepository = Mockito.mock(UserSelfieRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        photoSearchService = Mockito.mock(PhotoSearchService::class.java)
        emailService = Mockito.mock(EmailService::class.java)
    }

    @Test
    fun `a match claims once and sends the email`() {
        val alert = stubAlert(matchTotal = 3L)
        stubClaim(alert, won = true)

        notifier().notifyIfMatched(alert.id)

        Mockito.verify(emailService)
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
    }

    @Test
    fun `losing the claim to a concurrent sweep sends nothing`() {
        val alert = stubAlert(matchTotal = 3L)
        stubClaim(alert, won = false)

        notifier().notifyIfMatched(alert.id)

        Mockito.verify(emailService, Mockito.never())
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
    }

    @Test
    fun `a failed send releases the claim so the next sweep retries`() {
        val alert = stubAlert(matchTotal = 3L)
        stubClaim(alert, won = true)
        Mockito.doThrow(RuntimeException("resend down")).`when`(emailService)
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())

        notifier().notifyIfMatched(alert.id)

        Mockito.verify(alertRepository).releaseNotify(alert.id)
    }

    @Test
    fun `a successful send keeps the claim`() {
        val alert = stubAlert(matchTotal = 3L)
        stubClaim(alert, won = true)

        notifier().notifyIfMatched(alert.id)

        Mockito.verify(alertRepository, Mockito.never()).releaseNotify(anyArg())
    }

    @Test
    fun `no match never claims and never sends`() {
        val alert = stubAlert(matchTotal = 0L)

        notifier().notifyIfMatched(alert.id)

        assertNotNull(alert.lastCheckedAt)
        Mockito.verify(alertRepository, Mockito.never()).claimNotify(anyArg(), anyArg())
        Mockito.verify(emailService, Mockito.never())
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
        Mockito.verify(photoSearchService).searchByFace(
            anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), eqArg(false),
        )
    }

    @Test
    fun `an already-notified alert returns before searching`() {
        val alert = EventPhotoAlert(eventId = UUID.randomUUID(), userId = UUID.randomUUID())
        alert.notifiedAt = OffsetDateTime.now()
        Mockito.`when`(alertRepository.findById(alert.id)).thenReturn(Optional.of(alert))

        notifier().notifyIfMatched(alert.id)

        Mockito.verifyNoInteractions(photoSearchService)
        Mockito.verify(emailService, Mockito.never())
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
    }

    @Test
    fun `ai-api disabled returns before searching`() {
        val alert = EventPhotoAlert(eventId = UUID.randomUUID(), userId = UUID.randomUUID())
        Mockito.`when`(alertRepository.findById(alert.id)).thenReturn(Optional.of(alert))

        notifier(AiApiProperties(enabled = false)).notifyIfMatched(alert.id)

        Mockito.verifyNoInteractions(photoSearchService)
    }

    @Test
    fun `storage search and email run outside transactions while the claim runs inside one`() {
        val tx = TxTracker()
        val alert = stubAlert(matchTotal = 3L)
        var openAtStorage: Boolean? = null
        var openAtSearch: Boolean? = null
        var openAtClaim: Boolean? = null
        var openAtSend: Boolean? = null
        Mockito.`when`(storageService.getBytes(anyArg())).thenAnswer {
            openAtStorage = tx.open
            byteArrayOf(1, 2, 3)
        }
        Mockito.`when`(
            photoSearchService.searchByFace(
                anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(),
            ),
        ).thenAnswer {
            openAtSearch = tx.open
            PaginatedResponse(emptyList<PhotoDto>(), 3L, 0, 1)
        }
        Mockito.`when`(alertRepository.claimNotify(eqArg(alert.id), anyArg())).thenAnswer {
            openAtClaim = tx.open
            1
        }
        Mockito.doAnswer {
            openAtSend = tx.open
            null
        }.`when`(emailService)
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())

        notifier(transactionTemplate = tx.template).notifyIfMatched(alert.id)

        assertEquals(false, openAtStorage, "storage must not be read while a DB connection is held")
        assertEquals(false, openAtSearch, "face search must not run while a DB connection is held")
        assertEquals(true, openAtClaim, "the claim is a @Modifying UPDATE and must be transactional")
        assertEquals(false, openAtSend, "email must not be sent while a DB connection is held")
    }

    @Test
    fun `a failed send releases the claim in one additional transaction`() {
        val tx = TxTracker()
        val alert = stubAlert(matchTotal = 3L)
        stubClaim(alert, won = true)
        var executionsAtSend: Int? = null
        Mockito.doAnswer {
            executionsAtSend = tx.executions
            throw RuntimeException("resend down")
        }.`when`(emailService)
            .sendEventPhotosReadyEmail(anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyInt())
        var openAtRelease: Boolean? = null
        Mockito.`when`(alertRepository.releaseNotify(eqArg(alert.id))).thenAnswer {
            openAtRelease = tx.open
            1
        }

        notifier(transactionTemplate = tx.template).notifyIfMatched(alert.id)

        assertEquals(executionsAtSend!! + 1, tx.executions, "failure must add exactly one release transaction")
        assertEquals(true, openAtRelease, "releaseNotify must run inside the additional transaction")
    }

    @Test
    fun `the notifier entrypoint has no method-wide transaction`() {
        val method = EventPhotosReadyNotifier::class.java.getDeclaredMethod("notifyIfMatched", UUID::class.java)

        assertNull(
            AnnotationTransactionAttributeSource().getTransactionAttribute(
                method,
                EventPhotosReadyNotifier::class.java,
            ),
            "Spring must not wrap storage, face search, or email in an outer transaction",
        )
    }

    // ─── fixtures ─────────────────────────────────────────────────────────

    private fun notifier(
        props: AiApiProperties = AiApiProperties(enabled = true),
        transactionTemplate: TransactionTemplate = testTransactionTemplate(),
    ) =
        EventPhotosReadyNotifier(
            alertRepository,
            eventRepository,
            userRepository,
            userSelfieRepository,
            storageService,
            photoSearchService,
            props,
            emailService,
            transactionTemplate,
        )

    private class TxTracker {
        var open = false
        var executions = 0
        val template: TransactionTemplate = TransactionTemplate(
            object : AbstractPlatformTransactionManager() {
                override fun doGetTransaction(): Any = Any()
                override fun doBegin(transaction: Any, definition: TransactionDefinition) {
                    executions++
                    open = true
                }

                override fun doCommit(status: DefaultTransactionStatus) {
                    open = false
                }

                override fun doRollback(status: DefaultTransactionStatus) {
                    open = false
                }
            },
        )
    }

    private fun stubClaim(alert: EventPhotoAlert, won: Boolean) {
        Mockito.`when`(alertRepository.claimNotify(eqArg(alert.id), anyArg()))
            .thenReturn(if (won) 1 else 0)
    }

    // An opt-in that clears every precondition ahead of the claim; matchTotal
    // controls whether the event has photos of the runner yet (0 = not ready).
    private fun stubAlert(matchTotal: Long): EventPhotoAlert {
        val userId = UUID.randomUUID()
        val selfieId = UUID.randomUUID()
        val alert = EventPhotoAlert(eventId = UUID.randomUUID(), userId = userId, selfieId = selfieId)
        val selfie = UserSelfie(id = selfieId, userId = userId, s3Key = "selfies/$userId/$selfieId.jpg")
        val event = Event(
            slug = "cebu-marathon",
            name = "Cebu Marathon",
            date = LocalDate.of(2026, 1, 11),
            location = "Cebu City, Cebu",
            status = EventStatus.ACTIVE,
        )
        val user = User(email = "runner@test.local", passwordHash = "x", name = "Runner", role = Role.RUNNER)

        Mockito.`when`(alertRepository.findById(alert.id)).thenReturn(Optional.of(alert))
        Mockito.`when`(eventRepository.findById(alert.eventId)).thenReturn(Optional.of(event))
        Mockito.`when`(userSelfieRepository.findByIdAndUserId(selfieId, userId)).thenReturn(selfie)
        Mockito.`when`(storageService.getBytes(anyArg())).thenReturn(byteArrayOf(1, 2, 3))
        Mockito.`when`(
            photoSearchService.searchByFace(
                anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), anyArg(), Mockito.anyBoolean(),
            ),
        ).thenReturn(PaginatedResponse(emptyList<PhotoDto>(), matchTotal, 0, 1))
        Mockito.`when`(userRepository.findById(userId)).thenReturn(Optional.of(user))
        return alert
    }

    // Mockito.eq returns a platform type; wrapping keeps Kotlin's null-check off a
    // non-null parameter. Same shape as OrderReceiptEmailClaimTest.
    private fun <T> eqArg(value: T): T = Mockito.eq(value) ?: value

    private fun <T> anyArg(): T = Mockito.any()
}
