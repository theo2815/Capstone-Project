package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.dto.photographer.PatchPayoutRequest
import com.quickpitik.dto.photographer.PayoutAccountDto
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutAccountAudit
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.entity.FlagTargetKind
import com.quickpitik.entity.RepeatMatchKind
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerReasonCode
import com.quickpitik.entity.SystemFlagReason
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountAuditRepository
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.repository.PayoutCycleRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.SystemFlagService
import com.quickpitik.service.image.ImageDecode
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationEventPublisher
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionSynchronization
import org.springframework.transaction.support.TransactionSynchronizationManager
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.OffsetDateTime
import java.util.UUID
import javax.imageio.ImageIO

@Service
@Transactional
class PayoutAccountService(
    private val payoutAccountRepository: PayoutAccountRepository,
    private val payoutAccountAuditRepository: PayoutAccountAuditRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
    private val eventPublisher: ApplicationEventPublisher,
    private val payoutCycleRepository: PayoutCycleRepository,
    private val userRepository: UserRepository,
    private val sellerDetailsRepository: SellerDetailsRepository,
    private val systemFlagService: SystemFlagService,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun list(userId: UUID): List<PayoutAccountDto> =
        payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(userId)
            .map { it.toDto(qrUrl = qrUrlFor(it), qrUploadedAt = it.createdAt) }

    fun create(userId: UUID, req: CreatePayoutRequest): PayoutAccountDto {
        userRepository.lockById(userId)
        val method = PayoutMethod.fromWire(req.method) ?: throw ValidationException(
            code = ErrorCodes.VALIDATION_ERROR,
            message = "method must be one of gcash|maya|gotyme",
            field = "method",
        )
        val accountNumber = sanitizeNumber(req.accountNumber)
        validateAccountNumber(method, accountNumber)
        val accountName = req.accountName.trim()
        validateAccountName(accountName)

        val isFirst = payoutAccountRepository.countByUserId(userId) == 0L
        val saved = payoutAccountRepository.save(
            PayoutAccount(
                userId = userId,
                method = method,
                accountNumber = accountNumber,
                accountName = accountName,
                isPrimary = isFirst,
            ),
        )
        recordChange(userId, "create", method, newNumber = accountNumber)
        flagIfNameDiffers(userId, saved)
        matchPayoutNumber(userId, saved)
        return saved.toDto(qrUrl = null, qrUploadedAt = null)
    }

    fun patch(userId: UUID, id: UUID, req: PatchPayoutRequest): PayoutAccountDto {
        userRepository.lockById(userId)
        val account = payoutAccountRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout account not found")
        val oldNumber = account.accountNumber
        val oldName = account.accountName
        req.accountNumber?.let {
            val sanitized = sanitizeNumber(it)
            validateAccountNumber(account.method, sanitized)
            account.accountNumber = sanitized
        }
        req.accountName?.let {
            val trimmed = it.trim()
            validateAccountName(trimmed)
            account.accountName = trimmed
        }
        if (account.accountNumber != oldNumber) {
            val oldQr = account.qrS3Key
            account.qrS3Key = null
            deleteUnusedQrAfterCommit(oldQr)
        }
        val saved = payoutAccountRepository.save(account)
        recordChange(userId, "update", account.method, oldNumber = oldNumber, newNumber = saved.accountNumber)
        if (saved.accountName != oldName) flagIfNameDiffers(userId, saved)
        if (saved.accountNumber != oldNumber) matchPayoutNumber(userId, saved)
        return saved.toDto(qrUrl = qrUrlFor(saved), qrUploadedAt = saved.createdAt)
    }

    // AA-I5: a payout account checked against the seller details after submit. Runs under the
    // caller's user lock, never refuses the save, and leaves the 24 h cooldown to recordChange.
    private fun flagIfNameDiffers(userId: UUID, account: PayoutAccount) {
        val details = sellerDetailsToCheck(userId) ?: return
        if (NameMatch.same(details.legalName, account.accountName)) return
        systemFlagService.raise(
            FlagTargetKind.USER, userId, SystemFlagReason.PAYOUT_NAME_CHANGED,
            "${account.method.wire} ${maskNumber(account.accountNumber)}: holder name differs from the legal name",
        )
    }

    // A number shared with another account sends the record to manual review (D5, P1-N11), even
    // when it had passed; the revision bump makes a stale admin decision fail.
    private fun matchPayoutNumber(userId: UUID, account: PayoutAccount) {
        val details = sellerDetailsToCheck(userId) ?: return
        val others = payoutAccountRepository.findUserIdsByAccountNumbers(listOf(account.accountNumber), userId)
        if (others.isEmpty()) return
        val matches = SellerDetailsService.mergeMatches(details.repeatMatches, others.map { it to RepeatMatchKind.PAYOUT_NUMBER })
        val reasons = (details.reasonCodes - SellerReasonCode.PASSED).let {
            if (SellerReasonCode.REPEAT_ACCOUNT_MATCH in it) it else it + SellerReasonCode.REPEAT_ACCOUNT_MATCH
        }
        if (matches == details.repeatMatches && reasons == details.reasonCodes) return
        val moved = details.status != SellerDetailsStatus.MANUAL_REVIEW
        details.repeatMatches = matches
        details.reasonCodes = reasons
        details.status = SellerDetailsStatus.MANUAL_REVIEW
        details.revision += 1
        sellerDetailsRepository.save(details)
        if (moved) {
            systemFlagService.raise(
                FlagTargetKind.USER, userId, SystemFlagReason.REPEAT_ACCOUNT_MATCH,
                "${account.method.wire} ${maskNumber(account.accountNumber)} is also on account ${others.joinToString()}",
            )
        }
    }

    // Read across to the photographer's SellerDetails (rule 3); a rejected or missing one is left alone.
    private fun sellerDetailsToCheck(userId: UUID): SellerDetails? =
        sellerDetailsRepository.findById(userId).orElse(null)?.takeUnless { it.status == SellerDetailsStatus.REJECTED }

    fun delete(userId: UUID, id: UUID) {
        userRepository.lockById(userId)
        val account = payoutAccountRepository.findByIdAndUserId(id, userId) ?: return
        val wasPrimary = account.isPrimary
        val qrKey = account.qrS3Key
        val auditMethod = account.method
        val auditNumber = account.accountNumber
        payoutAccountRepository.delete(account)
        // Force the unique constraint to register the delete before we set a
        // new primary — otherwise the partial unique index would block the
        // promotion in the same TX.
        payoutAccountRepository.flush()
        if (wasPrimary) {
            val remaining = payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(userId)
            remaining.firstOrNull()?.let { next ->
                next.isPrimary = true
                payoutAccountRepository.save(next)
            }
        }
        deleteUnusedQrAfterCommit(qrKey)
        recordChange(userId, "delete", auditMethod, oldNumber = auditNumber)
    }

    // Atomic clear-then-set so the partial unique index is satisfied at TX
    // commit time. Returns the full canonical list so the FE can refresh the
    // primary chip across all rows from a single response.
    fun setPrimary(userId: UUID, id: UUID): List<PayoutAccountDto> {
        userRepository.lockById(userId)
        val target = payoutAccountRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout account not found")
        if (target.isPrimary) {
            return list(userId)
        }
        val all = payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(userId)
        all.forEach { if (it.isPrimary) it.isPrimary = false }
        payoutAccountRepository.saveAll(all)
        payoutAccountRepository.flush()
        target.isPrimary = true
        payoutAccountRepository.save(target)
        recordChange(userId, "set_primary", target.method, newNumber = target.accountNumber)
        return list(userId)
    }

    fun uploadQr(userId: UUID, id: UUID, bytes: ByteArray, contentType: String?): PayoutAccountDto {
        userRepository.lockById(userId)
        val account = payoutAccountRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout account not found")
        val mime = (contentType ?: "").lowercase()
        if (mime !in SUPPORTED_TYPES) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "QR must be jpeg, png, or webp",
                field = "file",
            )
        }
        if (bytes.isEmpty()) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "file is empty",
                field = "file",
            )
        }
        if (bytes.size > MAX_QR_BYTES) {
            throw ApiException(
                status = HttpStatus.PAYLOAD_TOO_LARGE,
                code = ErrorCodes.PAYLOAD_TOO_LARGE,
                message = "QR upload must be ≤ 8 MB",
                field = "file",
            )
        }
        if (ImagePixelGuard.exceedsPixelBudget(bytes)) {
            throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "Image dimensions exceed the supported maximum.",
                field = "qr",
            )
        }
        val processed = scaleToLongEdgePng(bytes, MAX_QR_LONG_EDGE)
            ?: throw ValidationException(
                code = ErrorCodes.UNSUPPORTED_MEDIA_TYPE,
                message = "image cannot be decoded",
                field = "file",
            )
        val previousKey = account.qrS3Key
        val newKey = "photographers/$userId/payouts/$id/${UUID.randomUUID()}.png"
        storageService.put(newKey, processed, "image/png")
        account.qrS3Key = newKey
        val saved = payoutAccountRepository.save(account)
        recordChange(userId, "qr", account.method, newNumber = account.accountNumber)
        // A payout's immutable QR stays available even after account replacement
        // or deletion. Keys are unique per upload and retained for its audit trail.
        deleteUnusedQrAfterCommit(previousKey)
        return saved.toDto(
            qrUrl = qrUrlFor(saved),
            qrUploadedAt = OffsetDateTime.now(),
        )
    }

    private fun qrUrlFor(account: PayoutAccount): String? = account.qrS3Key?.let {
        storageService.presignedGetUrl(it, storageProperties.presignedTtl.cover)
    }

    private fun deleteUnusedQrAfterCommit(key: String?) {
        if (key == null || payoutCycleRepository.existsByAccountSnapshotQrKey(key)) return
        val cleanup = {
            runCatching { storageService.delete(key) }
                .onFailure { log.warn("payout qr delete of {} failed: {}", key, it.message) }
            Unit
        }
        if (TransactionSynchronizationManager.isSynchronizationActive()) {
            TransactionSynchronizationManager.registerSynchronization(object : TransactionSynchronization {
                override fun afterCommit() = cleanup()
            })
        } else cleanup()
    }

    private fun sanitizeNumber(raw: String): String = raw.filter { it.isDigit() }

    // Append-only forensic trail of payout-destination changes. The most recent
    // row also starts the anti-fraud cooldown that PayoutRequestService enforces,
    // so EVERY mutation (incl. set-primary and QR) records one. Numbers are masked
    // — this is a trail, not a second copy of the sensitive number.
    private fun recordChange(
        userId: UUID,
        action: String,
        method: PayoutMethod?,
        oldNumber: String? = null,
        newNumber: String? = null,
    ) {
        val maskedOld = oldNumber?.let(::maskNumber)
        val maskedNew = newNumber?.let(::maskNumber)
        payoutAccountAuditRepository.save(
            PayoutAccountAudit(
                userId = userId,
                action = action,
                method = method?.wire,
                accountMaskedOld = maskedOld,
                accountMaskedNew = maskedNew,
            ),
        )
        // Owner alert (AFTER_COMMIT + async) so a compromised session's change is
        // visible to the real owner during the cooldown. Never carries the raw number.
        eventPublisher.publishEvent(
            PayoutAccountChangedEvent(
                userId = userId,
                action = action,
                method = method?.wire,
                maskedAccount = maskedNew ?: maskedOld,
            ),
        )
    }

    private fun maskNumber(n: String): String =
        if (n.length <= 4) "••••" else "••••" + n.takeLast(4)

    // Per plan F.2 line 226: gcash/maya = 11 digits starting 09; gotyme = 10–16.
    private fun validateAccountNumber(method: PayoutMethod, sanitized: String) {
        when (method) {
            PayoutMethod.GCASH, PayoutMethod.MAYA -> {
                if (sanitized.length != 11 || !sanitized.startsWith("09")) {
                    throw ValidationException(
                        code = ErrorCodes.VALIDATION_ERROR,
                        message = "${method.wire} accountNumber must be 11 digits starting with 09",
                        field = "accountNumber",
                    )
                }
            }
            PayoutMethod.GOTYME -> {
                if (sanitized.length !in 10..16) {
                    throw ValidationException(
                        code = ErrorCodes.VALIDATION_ERROR,
                        message = "gotyme accountNumber must be 10-16 digits",
                        field = "accountNumber",
                    )
                }
            }
        }
    }

    private fun validateAccountName(name: String) {
        if (name.isEmpty() || name.length > 64) {
            throw ValidationException(
                code = ErrorCodes.VALIDATION_ERROR,
                message = "accountName must be 1-64 chars",
                field = "accountName",
            )
        }
    }

    private fun scaleToLongEdgePng(input: ByteArray, maxLongEdge: Int): ByteArray? {
        val source = ImageDecode.readOrNull(input) ?: return null
        val srcW = source.width
        val srcH = source.height
        val longEdge = maxOf(srcW, srcH)
        val scale = if (longEdge > maxLongEdge) maxLongEdge.toDouble() / longEdge else 1.0
        val targetW = (srcW * scale).toInt().coerceAtLeast(1)
        val targetH = (srcH * scale).toInt().coerceAtLeast(1)
        // Preserve alpha — QR codes are commonly transparent PNGs and matte
        // backgrounds break scanability on some banking apps.
        val resized = BufferedImage(targetW, targetH, BufferedImage.TYPE_INT_ARGB)
        val g = resized.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(source, 0, 0, targetW, targetH, null)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        ImageIO.write(resized, "png", out)
        return out.toByteArray()
    }

    private companion object {
        val SUPPORTED_TYPES = setOf("image/jpeg", "image/jpg", "image/png", "image/webp")
        const val MAX_QR_LONG_EDGE = 800
        const val MAX_QR_BYTES = 8L * 1024 * 1024
    }
}
