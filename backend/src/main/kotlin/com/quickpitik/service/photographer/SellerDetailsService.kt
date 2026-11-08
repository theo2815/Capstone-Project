package com.quickpitik.service.photographer

import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.common.PaginatedResponse
import com.quickpitik.common.PaginationParams
import com.quickpitik.config.StorageProperties
import com.quickpitik.dto.admin.AdminSellerDetailsDto
import com.quickpitik.dto.admin.AdminSellerDetailsRowDto
import com.quickpitik.dto.admin.SellerDetailsFilesDto
import com.quickpitik.dto.admin.toAdminDto
import com.quickpitik.dto.admin.toAdminRowDto
import com.quickpitik.dto.photographer.SellerDetailsDto
import com.quickpitik.dto.photographer.SellerDetailsForm
import com.quickpitik.dto.photographer.toDto
import com.quickpitik.entity.PhotographerMessageKind
import com.quickpitik.entity.SellerDetails
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.SellerIdType
import com.quickpitik.exception.ApiException
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ValidationException
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.admin.AdminDecisionLogService
import com.quickpitik.service.image.ExifOrientation
import com.quickpitik.service.image.ImageDecode
import com.quickpitik.service.image.ImagePixelGuard
import com.quickpitik.service.storage.StorageService
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.transaction.support.TransactionSynchronization
import org.springframework.transaction.support.TransactionSynchronizationManager
import org.springframework.transaction.support.TransactionTemplate
import org.springframework.web.multipart.MultipartFile
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.time.Clock
import java.time.OffsetDateTime
import java.util.UUID
import javax.imageio.ImageIO

/**
 * A photographer's seller details (RA 11967 Sec. 21(b), BIR RR 16-2023), required
 * before new uploads (PhotoUploadService gate). Only the photographer and admins
 * read a record; the files are private and never go to face search.
 *
 * Never log names, the address, the phone or file bytes: delete failures log the
 * storage key only, which holds no personal data.
 */
@Service
class SellerDetailsService(
    private val sellerDetailsRepository: SellerDetailsRepository,
    private val userRepository: UserRepository,
    private val storageService: StorageService,
    private val transactionTemplate: TransactionTemplate,
    private val clock: Clock,
    private val adminDecisionLogService: AdminDecisionLogService,
    private val storageProperties: StorageProperties,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun get(userId: UUID): SellerDetailsDto =
        sellerDetailsRepository.findById(userId).map { it.toDto() }.orElse(SellerDetailsDto.MISSING)

    /**
     * Validates and re-encodes every file before storing any, stores them, then
     * saves the record in a short transaction (no DB transaction is held while
     * decoding or talking to storage). Replaced files are deleted after the commit;
     * this request's files are deleted if anything fails before the commit or the
     * transaction rolls back, and kept when the commit's outcome is unknown.
     */
    fun submit(userId: UUID, form: SellerDetailsForm): SellerDetailsDto {
        val legalName = text(form.legalName, "legalName", 200)
        val idType = SellerIdType.fromWire(form.idType?.trim())
            ?: throw ValidationException("Choose one of the listed ID types.", field = "idType")
        val address = text(form.address, "address", 500)
        val phone = validPhone(form.phone)
            ?: throw ValidationException("Enter a Philippine phone number starting with +63 or 0.", field = "phone")
        val idFront = jpeg(required(sent(form.idFront), "idFront"), "idFront")
        val idBack = (if (idType == SellerIdType.PHILSYS) required(sent(form.idBack), "idBack") else sent(form.idBack))
            ?.let { jpeg(it, "idBack") }
        val selfie = jpeg(required(sent(form.selfie), "selfie"), "selfie")
        val birForm = sent(form.birForm)?.let { jpeg(it, "birForm") }

        val newKeys = mutableListOf<String>()
        fun store(bytes: ByteArray): String = "$PREFIX$userId/${UUID.randomUUID()}.jpg".also {
            newKeys += it
            storageService.put(it, bytes, "image/jpeg")
        }
        var outcome: Int? = null // how the transaction ended; null while it has not run
        val (saved, oldKeys) = try {
            val frontKey = store(idFront)
            val backKey = idBack?.let(::store)
            val selfieKey = store(selfie)
            val birKey = birForm?.let(::store)
            checkNotNull(transactionTemplate.execute {
                TransactionSynchronizationManager.registerSynchronization(object : TransactionSynchronization {
                    override fun afterCompletion(status: Int) { outcome = status }
                })
                userRepository.lockById(userId)
                val existing = sellerDetailsRepository.findById(userId).orElse(null)
                val now = OffsetDateTime.now(clock)
                val row = existing ?: SellerDetails(
                    userId = userId, legalName = legalName, idType = idType, idFrontKey = frontKey,
                    idBackKey = backKey, selfieKey = selfieKey, birFormKey = birKey, address = address,
                    phone = phone, phoneKey = phoneKey(phone), status = SellerDetailsStatus.MANUAL_REVIEW,
                    revision = 0, submittedAt = now,
                )
                val old = existing?.fileKeys().orEmpty()
                row.apply {
                    this.legalName = legalName
                    this.idType = idType
                    idFrontKey = frontKey
                    idBackKey = backKey
                    this.selfieKey = selfieKey
                    birFormKey = birKey
                    this.address = address
                    this.phone = phone
                    phoneKey = phoneKey(phone)
                    status = SellerDetailsStatus.MANUAL_REVIEW
                    reasonCodes = emptyList()
                    rejectionReason = null
                    reviewedAt = null
                    reviewedBy = null
                    submittedAt = now
                    revision += 1
                }
                sellerDetailsRepository.save(row) to old
            })
        } catch (error: Exception) {
            if (outcome == null || outcome == TransactionSynchronization.STATUS_ROLLED_BACK) {
                newKeys.forEach(::deleteQuietly)
            } else {
                // The commit may have landed (a lost acknowledgement), so the row may point to the new
                // files: keep them. shortcut: no sweep removes them if it did not; add one if this log appears.
                log.warn("seller details commit outcome unknown; kept new files {}", newKeys)
            }
            throw error
        }
        oldKeys.forEach(::deleteQuietly)
        return saved.toDto()
    }

    /** E3: the review queue, waiting records first. No links: an admin opens files one record at a time. */
    @Transactional(readOnly = true)
    fun adminList(status: String?, params: PaginationParams): PaginatedResponse<AdminSellerDetailsRowDto> {
        val filter = status?.let { wire ->
            SellerDetailsStatus.fromWire(wire)
                ?: throw ValidationException(
                    "status must be manual_review, auto_approved, approved or rejected (or the legacy submitted).",
                    field = "status",
                )
        }
        val page = sellerDetailsRepository.findForAdmin(filter, OffsetLimitPageable(params))
        val emails = userRepository.findAllById(page.content.map { it.userId }).associate { it.id to it.email }
        return PaginatedResponse.of(page.content.map { it.toAdminRowDto(emails[it.userId].orEmpty()) }, page.totalElements, params)
    }

    /** E4: one record with 5-minute file links; every view is logged (which files, never the links). */
    @Transactional
    fun adminView(adminId: UUID, userId: UUID): AdminSellerDetailsDto {
        val row = find(userId)
        val ttl = storageProperties.presignedTtl.photographerDownload
        // Presigning is local signing (no storage call), so it may run inside the transaction.
        fun link(key: String?, part: String) = key?.let { storageService.presignedDownloadUrl(it, ttl, "$part.jpg") }
        val files = SellerDetailsFilesDto(
            idFront = link(row.idFrontKey, "id_front")!!,
            idBack = link(row.idBackKey, "id_back"),
            selfie = link(row.selfieKey, "selfie")!!,
            birForm = link(row.birFormKey, "bir_form"),
        )
        val present = listOfNotNull("id_front", "id_back".takeIf { row.idBackKey != null }, "selfie", "bir_form".takeIf { row.birFormKey != null })
        adminDecisionLogService.logUserDecision(
            adminId = adminId, targetUserId = userId, decision = "seller_details_viewed",
            meta = mapOf("files" to present, "revision" to row.revision),
        )
        return row.toAdminDto(email(userId), files, OffsetDateTime.now(clock).plus(ttl))
    }

    /** E5: only a MANUAL_REVIEW record at the revision the admin opened. */
    @Transactional
    fun approve(adminId: UUID, userId: UUID, revision: Long): AdminSellerDetailsDto {
        val row = decide(userId, revision, SellerDetailsStatus.MANUAL_REVIEW)
        row.status = SellerDetailsStatus.APPROVED
        row.reviewedAt = OffsetDateTime.now(clock)
        row.reviewedBy = adminId
        sellerDetailsRepository.save(row)
        adminDecisionLogService.logUserDecision(
            adminId = adminId, targetUserId = userId, decision = "seller_details_approved", meta = mapOf("revision" to revision),
        )
        return row.toAdminDto(email(userId))
    }

    /** E6: from MANUAL_REVIEW, or AUTO_APPROVED / APPROVED (a fake found later); pauses uploads and tells the photographer why. */
    @Transactional
    fun reject(adminId: UUID, userId: UUID, revision: Long, reason: String): AdminSellerDetailsDto {
        val row = decide(
            userId, revision,
            SellerDetailsStatus.MANUAL_REVIEW, SellerDetailsStatus.AUTO_APPROVED, SellerDetailsStatus.APPROVED,
        )
        val trimmed = reason.trim()
        row.status = SellerDetailsStatus.REJECTED
        row.rejectionReason = trimmed
        row.reviewedAt = OffsetDateTime.now(clock)
        row.reviewedBy = adminId
        sellerDetailsRepository.save(row)
        val decision = adminDecisionLogService.logUserDecision(
            adminId = adminId, targetUserId = userId, decision = "seller_details_rejected",
            reason = trimmed, meta = mapOf("revision" to revision),
        )
        adminDecisionLogService.pushMessage(
            photographerId = userId,
            kind = PhotographerMessageKind.ADMIN_MESSAGE,
            title = "Seller details not accepted",
            body = "Your seller details were not accepted. Reason: ${trimmed.trimEnd('.')}. New uploads are paused until you send them " +
                "again in your photographer settings on quickpitik.com.",
            sourceAdminId = adminId,
            sourceDecisionId = decision.id,
        )
        return row.toAdminDto(email(userId))
    }

    private fun find(userId: UUID): SellerDetails = sellerDetailsRepository.findById(userId)
        .orElseThrow { NotFoundException("Seller details not found", ErrorCodes.NOT_FOUND) }

    private fun email(userId: UUID): String = userRepository.findById(userId).map { it.email }.orElse("")

    /** Locks the user row (as submit does), so a resubmission and a decision never interleave. */
    private fun decide(userId: UUID, revision: Long, vararg allowed: SellerDetailsStatus): SellerDetails {
        userRepository.lockById(userId)
        val row = find(userId)
        if (row.status !in allowed) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION,
                "Seller details are ${row.status.toWire()} and cannot be decided this way.")
        }
        if (row.revision != revision) {
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.INVALID_STATE_TRANSITION,
                "Seller details have changed. Reload before deciding.", field = "revision")
        }
        return row
    }

    private fun deleteQuietly(key: String) {
        runCatching { storageService.delete(key) }
            .onFailure { log.warn("seller details file delete of {} failed: {}", key, it.message) }
    }

    private fun text(raw: String?, field: String, max: Int): String {
        val value = raw?.trim().orEmpty()
        if (value.isEmpty() || value.length > max) {
            throw ValidationException("$field must be 1 to $max characters.", field = field)
        }
        return value
    }

    // Spring binds a blank browser file input as an empty part, not null: treat it as not sent.
    private fun sent(file: MultipartFile?): MultipartFile? = file?.takeUnless { it.isEmpty }

    private fun required(file: MultipartFile?, field: String): MultipartFile =
        file ?: throw ValidationException("$field is required.", field = field)

    /** Checks one uploaded photo and returns it upright, at most 2400 px, as a fresh JPEG (no EXIF, no GPS). */
    private fun jpeg(file: MultipartFile, field: String): ByteArray {
        if ((file.contentType ?: "").lowercase() !in SUPPORTED_TYPES) {
            throw ValidationException("$field must be a JPEG, PNG or WebP image.", ErrorCodes.UNSUPPORTED_MEDIA_TYPE, field)
        }
        if (file.size > MAX_FILE_BYTES) {
            throw ApiException(HttpStatus.PAYLOAD_TOO_LARGE, ErrorCodes.PAYLOAD_TOO_LARGE, "$field must be 5 MB or smaller.", field)
        }
        val bytes = file.bytes
        if (ImagePixelGuard.exceedsPixelBudget(bytes)) {
            throw ValidationException("Image dimensions exceed the supported maximum.", ErrorCodes.UNSUPPORTED_MEDIA_TYPE, field)
        }
        val source = ImageDecode.readOrNull(bytes)
            ?: throw ValidationException("$field cannot be read as an image.", ErrorCodes.UNSUPPORTED_MEDIA_TYPE, field)
        val upright = ExifOrientation.apply(source, ExifOrientation.read(bytes))
        val scale = minOf(1.0, MAX_LONG_EDGE.toDouble() / maxOf(upright.width, upright.height))
        val width = (upright.width * scale).toInt().coerceAtLeast(1)
        val height = (upright.height * scale).toInt().coerceAtLeast(1)
        val out = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        val g = out.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC)
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY)
            g.drawImage(upright, 0, 0, width, height, null)
        } finally {
            g.dispose()
        }
        return ByteArrayOutputStream().also { ImageIO.write(out, "jpeg", it) }.toByteArray()
    }

    companion object {
        const val PREFIX = "seller-details/"
        const val MAX_FILE_BYTES = 5L * 1024 * 1024
        private const val MAX_LONG_EDGE = 2400
        private val SUPPORTED_TYPES = setOf("image/jpeg", "image/png", "image/webp")
        private val PHONE_SEPARATORS = Regex("[\\s.()\\-]")
        private val PHONE_SHAPE = Regex("\\+?\\d+")

        /** The § 3.4 phone rule: returns the number as typed (trimmed) when valid, else null. */
        fun validPhone(raw: String?): String? {
            val typed = raw?.trim() ?: return null
            val compact = typed.replace(PHONE_SEPARATORS, "")
            val digits = compact.count { it.isDigit() }
            val ok = typed.length <= 30 && PHONE_SHAPE.matches(compact) &&
                (compact.startsWith("+63") || compact.startsWith("0")) && digits in 10..13
            return typed.takeIf { ok }
        }

        /** The last 10 digits of a valid phone, for repeat-account matching (the same rule as V94's backfill). */
        fun phoneKey(phone: String): String = phone.filter(Char::isDigit).takeLast(10)
    }
}
