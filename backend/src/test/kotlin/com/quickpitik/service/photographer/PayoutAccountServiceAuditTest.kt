package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.dto.photographer.PatchPayoutRequest
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.RepeatMatch
import com.quickpitik.entity.RepeatMatchKind
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.storage.StorageService
import java.time.OffsetDateTime
import java.util.Optional
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.mockito.ArgumentCaptor
import org.mockito.Mockito
import org.springframework.context.ApplicationEventPublisher
import java.util.Base64
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

// Phase 3a — every payout-account change writes an append-only audit row, which
// also drives the post-change payout cooldown. This covers the create path;
// patch/delete/set-primary/qr call the same recordChange().
class PayoutAccountServiceAuditTest {

    private lateinit var payoutAccountRepository: PayoutAccountRepository
    private lateinit var payoutAccountAuditRepository: PayoutAccountAuditRepository
    private lateinit var storageService: StorageService
    private lateinit var storageProperties: StorageProperties
    private lateinit var eventPublisher: ApplicationEventPublisher

    private val userId = UUID.randomUUID()

    private fun <T> anyArg(): T = Mockito.any()

    @BeforeEach
    fun setUp() {
        payoutAccountRepository = Mockito.mock(PayoutAccountRepository::class.java)
        payoutAccountAuditRepository = Mockito.mock(PayoutAccountAuditRepository::class.java)
        storageService = Mockito.mock(StorageService::class.java)
        storageProperties = Mockito.mock(StorageProperties::class.java)
        eventPublisher = Mockito.mock(ApplicationEventPublisher::class.java)
    }

    private fun service() = PayoutAccountService(
        payoutAccountRepository,
        payoutAccountAuditRepository,
        storageService,
        storageProperties,
        eventPublisher,
        Mockito.mock(com.quickpitik.repository.PayoutCycleRepository::class.java),
        Mockito.mock(com.quickpitik.repository.UserRepository::class.java),
        sellerDetailsRepository,
        flags,
    )

    private val sellerDetailsRepository: SellerDetailsRepository = Mockito.mock(SellerDetailsRepository::class.java)
    private val flags: SystemFlagService = Mockito.mock(SystemFlagService::class.java)
    private val other = UUID.randomUUID()

    // AA-I5: the payout hook. Seller details with legal name "Juan Dela Cruz".
    private fun details(status: SellerDetailsStatus, vararg reasons: SellerReasonCode) = SellerDetails(
        userId = userId, legalName = "Juan Dela Cruz", idType = SellerIdType.PASSPORT, idFrontKey = "k/f", idBackKey = null,
        selfieKey = "k/s", birFormKey = null, address = "Cebu", phone = "09170000000", phoneKey = "9170000000",
        status = status, revision = 3, submittedAt = OffsetDateTime.parse("2026-11-01T00:00:00Z"),
    ).apply { reasonCodes = reasons.toList() }.also {
        Mockito.`when`(sellerDetailsRepository.findById(userId)).thenReturn(Optional.of(it))
    }

    private fun create(name: String = "Juan Dela Cruz") {
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }
        service().create(userId, CreatePayoutRequest(method = "gcash", accountNumber = "0917-123-4567", accountName = name))
    }

    private fun sharedNumber() = Mockito.`when`(payoutAccountRepository.findUserIdsByAccountNumbers(listOf("09171234567"), userId))
        .thenReturn(listOf(other))

    private fun raisedReasons(): List<Any?> =
        Mockito.mockingDetails(flags).invocations.filter { it.method.name == "raise" }.map { it.arguments[2] }

    @Test
    fun `a new account under another holder name raises a payout name flag`() {
        details(SellerDetailsStatus.APPROVED, SellerReasonCode.PASSED)

        create(name = "Maria Santos")

        Mockito.verify(flags).raise(
            FlagTargetKind.USER, userId, SystemFlagReason.PAYOUT_NAME_CHANGED,
            "gcash ••••4567: holder name differs from the legal name", null,
        )
    }

    @Test
    fun `a matching name raises none`() {
        details(SellerDetailsStatus.MANUAL_REVIEW, SellerReasonCode.DECLINED)

        create(name = "JUAN A. DELA CRUZ")

        assertEquals(emptyList(), raisedReasons())
    }

    // Two accounts saving one number at once take turns, so the second sees the first. Taken even with no
    // seller details: that account's number must still be seen by a same-moment save on another account.
    @Test
    fun `the payout number lock is taken before the repeat lookup`() {
        details(SellerDetailsStatus.APPROVED, SellerReasonCode.PASSED)

        create()

        val order = Mockito.inOrder(payoutAccountRepository)
        order.verify(payoutAccountRepository).lockAccountNumber("09171234567")
        order.verify(payoutAccountRepository).findUserIdsByAccountNumbers(listOf("09171234567"), userId)
    }

    @Test
    fun `a save with no seller details still takes the payout number lock`() {
        create()

        Mockito.verify(payoutAccountRepository).lockAccountNumber("09171234567")
    }

    @Test
    fun `no seller details raises none`() {
        sharedNumber()

        create(name = "Maria Santos")

        Mockito.verifyNoInteractions(flags)
        Mockito.verify(sellerDetailsRepository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    @Test
    fun `a number-only change raises no name flag`() {
        details(SellerDetailsStatus.APPROVED, SellerReasonCode.PASSED)
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09170000001", accountName = "Maria Santos")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().patch(userId, account.id, PatchPayoutRequest(accountNumber = "09171234567"))

        assertEquals(emptyList(), raisedReasons())
        // The new number is still matched.
        Mockito.verify(payoutAccountRepository).findUserIdsByAccountNumbers(listOf("09171234567"), userId)
    }

    @Test
    fun `a rename to another holder name raises a payout name flag`() {
        details(SellerDetailsStatus.APPROVED, SellerReasonCode.PASSED)
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567", accountName = "Juan Dela Cruz")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().patch(userId, account.id, PatchPayoutRequest(accountName = "Maria Santos"))

        assertEquals(listOf<Any?>(SystemFlagReason.PAYOUT_NAME_CHANGED), raisedReasons())
        // A name-only change runs no number match.
        Mockito.verify(payoutAccountRepository, Mockito.never()).findUserIdsByAccountNumbers(anyArg(), anyArg())
    }

    @Test
    fun `rejected seller details raise none`() {
        details(SellerDetailsStatus.REJECTED)
        sharedNumber()

        create(name = "Maria Santos")

        Mockito.verifyNoInteractions(flags)
        Mockito.verify(sellerDetailsRepository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    @Test
    fun `a payout number added after submit puts a manual review record on REPEAT_ACCOUNT_MATCH`() {
        val row = details(SellerDetailsStatus.MANUAL_REVIEW, SellerReasonCode.DECLINED)
        sharedNumber()

        create()

        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.DECLINED, SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.PAYOUT_NUMBER))), row.repeatMatches)
        assertEquals(4L, row.revision)
        Mockito.verify(sellerDetailsRepository).save(row)
        assertEquals(emptyList(), raisedReasons(), "already in the admin queue: no extra flag")
    }

    @Test
    fun `a payout number shared with another account moves an approved record to manual review and raises REPEAT_ACCOUNT_MATCH`() {
        val row = details(SellerDetailsStatus.APPROVED, SellerReasonCode.DECLINED)
        sharedNumber()

        create()

        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.DECLINED, SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
        assertEquals(listOf(RepeatMatch(other, listOf(RepeatMatchKind.PAYOUT_NUMBER))), row.repeatMatches)
        assertEquals(4L, row.revision, "a stale admin action fails")
        Mockito.verify(flags).raise(
            FlagTargetKind.USER, userId, SystemFlagReason.REPEAT_ACCOUNT_MATCH, "gcash ••••4567 is also on account $other", null,
        )
    }

    @Test
    fun `an auto approved record drops PASSED when a shared number moves it`() {
        val row = details(SellerDetailsStatus.AUTO_APPROVED, SellerReasonCode.PASSED)
        sharedNumber()

        create()

        assertEquals(SellerDetailsStatus.MANUAL_REVIEW, row.status)
        assertEquals(listOf(SellerReasonCode.REPEAT_ACCOUNT_MATCH), row.reasonCodes)
    }

    @Test
    fun `a payout number already recorded changes nothing`() {
        val row = details(SellerDetailsStatus.MANUAL_REVIEW, SellerReasonCode.REPEAT_ACCOUNT_MATCH)
            .apply { repeatMatches = listOf(RepeatMatch(other, listOf(RepeatMatchKind.PAYOUT_NUMBER))) }
        sharedNumber()

        create()

        assertEquals(3L, row.revision)
        Mockito.verify(sellerDetailsRepository, Mockito.never()).save(anyArg<SellerDetails>())
    }

    @Test
    fun `a payout number match never refuses the payout account save`() {
        details(SellerDetailsStatus.APPROVED, SellerReasonCode.PASSED)
        sharedNumber()
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        val dto = service().create(userId, CreatePayoutRequest(method = "gcash", accountNumber = "09171234567", accountName = "Juan Dela Cruz"))

        assertEquals("Juan Dela Cruz", dto.accountName)
        Mockito.verify(payoutAccountRepository).save(anyArg<PayoutAccount>())
    }

    @Test
    fun `creating a payout account writes a masked audit row`() {
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().create(
            userId,
            CreatePayoutRequest(method = "gcash", accountNumber = "09171234567", accountName = "Juan Dela Cruz"),
        )

        val captor = ArgumentCaptor.forClass(PayoutAccountAudit::class.java)
        Mockito.verify(payoutAccountAuditRepository).save(captor.capture())
        assertEquals("create", captor.value.action)
        assertEquals("gcash", captor.value.method)
        assertEquals("••••4567", captor.value.accountMaskedNew)
    }

    @Test
    fun `creating a payout account publishes an owner-alert event`() {
        Mockito.`when`(payoutAccountRepository.countByUserId(userId)).thenReturn(0L)
        Mockito.`when`(payoutAccountRepository.save(anyArg<PayoutAccount>())).thenAnswer { it.getArgument(0) }

        service().create(
            userId,
            CreatePayoutRequest(method = "gcash", accountNumber = "09171234567", accountName = "Juan Dela Cruz"),
        )

        val captor = ArgumentCaptor.forClass(PayoutAccountChangedEvent::class.java)
        Mockito.verify(eventPublisher).publishEvent(captor.capture())
        assertEquals("create", captor.value.action)
        assertEquals("••••4567", captor.value.maskedAccount)
    }
    @Test
    fun `truncated WebP QR is rejected before storage or account writes`() {
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567", accountName = "Juan Dela Cruz", qrS3Key = "previous.png")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)
        val bytes = Base64.getDecoder().decode("UklGRjIAAABXRUJQVlA4ICYAAAA=")

        val ex = assertFailsWith<ValidationException> {
            service().uploadQr(userId, account.id, bytes, "image/webp")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, ex.code)
        assertEquals("file", ex.field)
        assertEquals("previous.png", account.qrS3Key)
        Mockito.verifyNoInteractions(storageService, payoutAccountAuditRepository, eventPublisher)
        Mockito.verify(payoutAccountRepository, Mockito.never()).save(Mockito.any(PayoutAccount::class.java))
    }

    @Test
    fun `uploading a QR over the pixel budget is rejected before storage or save`() {
        val account = PayoutAccount(userId = userId, method = PayoutMethod.GCASH, accountNumber = "09171234567", accountName = "Juan Dela Cruz")
        Mockito.`when`(payoutAccountRepository.findByIdAndUserId(account.id, userId)).thenReturn(account)

        val error = assertFailsWith<ValidationException> {
            service().uploadQr(userId, account.id, overBudgetPngHeader(), "image/png")
        }

        assertEquals(ErrorCodes.UNSUPPORTED_MEDIA_TYPE, error.code)
        assertEquals("Image dimensions exceed the supported maximum.", error.message)
        assertEquals("qr", error.field)
        Mockito.verifyNoInteractions(storageService)
        Mockito.verify(payoutAccountRepository, Mockito.never()).save(Mockito.any(PayoutAccount::class.java))
    }

    // 10,000 × 8,001 = 80,010,000 pixels, above ImagePixelGuard.MAX_PIXELS.
    private fun overBudgetPngHeader(): ByteArray = byteArrayOf(
        -119, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 39, 16, 0, 0, 31, 65, 8, 6, 0, 0, 0, -26, 16, 82, 64,
    )
}
