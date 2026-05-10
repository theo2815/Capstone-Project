package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.photographer.CreatePayoutRequest
import com.quickpitik.dto.photographer.PatchPayoutRequest
import com.quickpitik.dto.photographer.PayoutAccountDto
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.PayoutAccount
import com.quickpitik.entity.PayoutMethod
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.PayoutAccountRepository
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.time.OffsetDateTime
import java.util.UUID
import javax.imageio.ImageIO

@Service
@Transactional
class PayoutAccountService(
    private val payoutAccountRepository: PayoutAccountRepository,
    private val storageService: StorageService,
    private val storageProperties: StorageProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    @Transactional(readOnly = true)
    fun list(userId: UUID): List<PayoutAccountDto> =
        payoutAccountRepository.findAllByUserIdOrderByCreatedAtAsc(userId)
            .map { it.toDto(qrUrl = qrUrlFor(it), qrUploadedAt = it.createdAt) }

    fun create(userId: UUID, req: CreatePayoutRequest): PayoutAccountDto {
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
        return saved.toDto(qrUrl = null, qrUploadedAt = null)
    }

    fun patch(userId: UUID, id: UUID, req: PatchPayoutRequest): PayoutAccountDto {
        val account = payoutAccountRepository.findByIdAndUserId(id, userId)
            ?: throw NotFoundException(code = ErrorCodes.PAYOUT_NOT_FOUND, message = "Payout account not found")
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
        val saved = payoutAccountRepository.save(account)
        return saved.toDto(qrUrl = qrUrlFor(saved), qrUploadedAt = saved.createdAt)
    }

    fun delete(userId: UUID, id: UUID) {
        val account = payoutAccountRepository.findByIdAndUserId(id, userId) ?: return
        val wasPrimary = account.isPrimary
        val qrKey = account.qrS3Key
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
        if (qrKey != null) {
            runCatching { storageService.delete(qrKey) }
                .onFailure { log.warn("payout qr delete of {} failed: {}", qrKey, it.message) }
        }
    }

    // Atomic clear-then-set so the partial unique index is satisfied at TX
    // commit time. Returns the full canonical list so the FE can refresh the
    // primary chip across all rows from a single response.
    fun setPrimary(userId: UUID, id: UUID): List<PayoutAccountDto> {
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
        return list(userId)
    }

    fun uploadQr(userId: UUID, id: UUID, bytes: ByteArray, contentType: String?): PayoutAccountDto {
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
        if (previousKey != null && previousKey != newKey) {
            runCatching { storageService.delete(previousKey) }
                .onFailure { log.warn("payout qr delete of {} failed: {}", previousKey, it.message) }
        }
        return saved.toDto(
            qrUrl = qrUrlFor(saved),
            qrUploadedAt = OffsetDateTime.now(),
        )
    }

    private fun qrUrlFor(account: PayoutAccount): String? = account.qrS3Key?.let {
        storageService.presignedGetUrl(it, storageProperties.presignedTtl.cover)
    }

    private fun sanitizeNumber(raw: String): String = raw.filter { it.isDigit() }

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
        val source = ImageIO.read(ByteArrayInputStream(input)) ?: return null
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
