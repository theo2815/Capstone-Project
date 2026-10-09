package com.quickpitik.service.admin

import com.quickpitik.config.AdminActionProperties
import com.quickpitik.exception.NotFoundException
import com.quickpitik.exception.ConflictException
import com.quickpitik.exception.ValidationException
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.common.ErrorCodes
import com.quickpitik.common.IdempotencyKey
import com.quickpitik.dto.admin.AdminActionCommandDto
import com.quickpitik.dto.admin.toDto
import com.quickpitik.entity.AdminActionCommand
import com.quickpitik.entity.AdminActionCommandState
import com.quickpitik.entity.AdminActionOperation
import com.quickpitik.exception.ApiException
import com.quickpitik.repository.AdminActionCommandRepository
import org.hibernate.exception.ConstraintViolationException
import org.springframework.dao.DataIntegrityViolationException
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.PlatformTransactionManager
import org.springframework.transaction.TransactionDefinition
import org.springframework.transaction.support.TransactionTemplate
import java.security.MessageDigest
import java.time.Clock
import java.time.Instant
import java.util.HexFormat
import java.util.UUID

data class AdminActionResult(val resourceId: UUID?, val decisionId: UUID?, val messageId: UUID?)

@Service
class AdminActionCommandService(
    private val commands: AdminActionCommandRepository,
    txManager: PlatformTransactionManager,
    private val mapper: ObjectMapper,
    private val clock: Clock,
    private val properties: AdminActionProperties = AdminActionProperties(),
) {
    private val tx = TransactionTemplate(txManager).apply {
        propagationBehavior = TransactionDefinition.PROPAGATION_REQUIRES_NEW
    }

    // Another admin's receipt answers exactly like a missing one, so a key's existence is never revealed.
    // The code differs from an unmapped path's NOT_FOUND: clients release a stored key only on this answer.
    fun find(actorId: UUID, key: IdempotencyKey?): AdminActionCommandDto =
        key?.let { commands.findById(UUID.fromString(it.value)).orElse(null) }
            ?.takeIf { it.actorId == actorId }?.toDto()
            ?: throw NotFoundException("Admin request not found", ErrorCodes.ADMIN_COMMAND_NOT_FOUND)

    fun execute(key: IdempotencyKey?, actorId: UUID, operation: AdminActionOperation, targetId: UUID?,
        inputHash: String, authorize: () -> Unit, action: () -> AdminActionResult): AdminActionResult {
        val commandId = key?.let { UUID.fromString(it.value) }
        return try {
            tx.execute transaction@ {
                authorize()
                if (commandId == null) return@transaction action()
                commands.findById(commandId).orElse(null)?.let {
                    return@transaction replay(it, actorId, operation, targetId, inputHash)
                }
                val now = clock.instant()
                val receipt = AdminActionCommand(commandId = commandId, actorId = actorId,
                    operation = operation, targetId = targetId, inputHash = inputHash, claimedAt = now, updatedAt = now)
                commands.insertClaim(commandId, actorId, operation.name, targetId, inputHash, receipt.claimOwner, now)
                val result = action()
                receipt.resultResourceId = result.resourceId
                receipt.resultDecisionId = result.decisionId
                receipt.resultMessageId = result.messageId
                receipt.state = AdminActionCommandState.SUCCEEDED
                receipt.completedAt = clock.instant()
                receipt.updatedAt = receipt.completedAt!!
                commands.saveAndFlush(receipt)
                result
            }!!
        } catch (error: DataIntegrityViolationException) {
            val constraint = generateSequence<Throwable>(error) { it.cause }
                .filterIsInstance<ConstraintViolationException>().firstOrNull()?.constraintName
            if (commandId == null || constraint != "admin_action_commands_pkey") throw error
            // The losing PostgreSQL transaction is rolled back before the winner is read.
            tx.execute {
                authorize()
                val receipt = commands.findById(commandId).orElse(null) ?: throw error
                replay(receipt, actorId, operation, targetId, inputHash)
            }!!
        }
    }

    fun executeEvent(key: IdempotencyKey, actorId: UUID, operation: AdminActionOperation, targetId: UUID?,
        inputHash: String, authorize: () -> Unit, validate: () -> Unit, prepare: () -> String?,
        writeCover: (String, Boolean) -> Unit, deleteCover: (String) -> Unit,
        action: (UUID, String?) -> AdminActionResult): AdminActionResult {
        val commandId = UUID.fromString(key.value)
        val completed = tx.execute {
            authorize()
            commands.findForUpdate(commandId)?.let {
                ensureIdentity(it, actorId, operation, targetId, inputHash)
                if (it.state == AdminActionCommandState.SUCCEEDED) return@execute replay(it, actorId, operation, targetId, inputHash)
                requireRecoverable(it)
            }
            null
        }
        if (completed != null) return completed
        // Image validation/normalization runs outside both the claim and final write transactions.
        val digest = prepare()
        val owner = UUID.randomUUID()
        var recovery = false
        val receipt = tx.execute {
            authorize()
            val now = clock.instant()
            val eventId = targetId ?: UUID.randomUUID()
            commands.insertEventClaim(commandId, actorId, operation.name, targetId, inputHash, owner, now,
                now.plus(properties.lease), eventId, digest?.let { "events/$eventId/cover/$commandId.jpg" }, digest)
            val current = commands.findForUpdate(commandId)!!
            ensureIdentity(current, actorId, operation, targetId, inputHash)
            if (current.state == AdminActionCommandState.SUCCEEDED) return@execute current
            if (current.claimOwner == owner) {
                validate()
                return@execute current
            }
            val takeoverAt = requireRecoverable(current)
            if (current.processedCoverDigest != digest) throw recoveryRequired()
            validate()
            check(commands.takeOver(commandId, current.claimOwner, owner, takeoverAt, takeoverAt.plus(properties.lease)) == 1)
            recovery = true
            commands.findForUpdate(commandId)!!
        }!!
        if (receipt.state == AdminActionCommandState.SUCCEEDED) return replay(receipt, actorId, operation, targetId, inputHash)
        var coverWritten = false
        try {
            receipt.intendedStorageKey?.let { writeCover(it, recovery); coverWritten = true }
            return tx.execute {
                authorize()
                val current = commands.findForUpdate(commandId)!!
                ensureIdentity(current, actorId, operation, targetId, inputHash)
                if (current.state == AdminActionCommandState.SUCCEEDED) return@execute replay(current, actorId, operation, targetId, inputHash)
                if (current.claimOwner != owner) throw recoveryRequired()
                val result = action(current.intendedEventId!!, current.intendedStorageKey)
                current.resultResourceId = result.resourceId
                current.resultDecisionId = result.decisionId
                current.state = AdminActionCommandState.SUCCEEDED
                current.completedAt = clock.instant()
                current.updatedAt = current.completedAt!!
                commands.saveAndFlush(current)
                result
            }!!
        } catch (error: Exception) {
            val typed = error is ApiException || error is NotFoundException ||
                error is ConflictException || error is ValidationException
            // A typed error after the cover was written came from the final transaction, which rolled back:
            // nothing was saved and no event points at the new object, so the outcome is known.
            val failed = typed && coverWritten
            var orphan: String? = null
            tx.executeWithoutResult {
                val current = commands.findForUpdate(commandId)!!
                if (current.claimOwner == owner && current.state != AdminActionCommandState.SUCCEEDED) {
                    current.state = if (failed) AdminActionCommandState.FAILED else AdminActionCommandState.UNKNOWN
                    current.updatedAt = clock.instant()
                    current.leaseExpiresAt = current.updatedAt
                    commands.saveAndFlush(current)
                    if (failed) orphan = current.intendedStorageKey
                }
            }
            orphan?.let(deleteCover)
            if (typed) throw error
            throw recoveryRequired()
        }
    }

    private fun requireRecoverable(receipt: AdminActionCommand): Instant {
        if (receipt.state == AdminActionCommandState.FAILED) throw ApiException(HttpStatus.CONFLICT,
            ErrorCodes.CONFLICT, "Command failed; this key cannot execute another action")
        if (receipt.leaseExpiresAt == null) throw recoveryRequired()
        val now = clock.instant()
        if (receipt.leaseExpiresAt!! > now)
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.IDEMPOTENT_REPLAY_IN_PROGRESS, "Command is still in progress")
        return now
    }

    private fun recoveryRequired() = ApiException(HttpStatus.CONFLICT,
        ErrorCodes.ADMIN_COMMAND_RECOVERY_REQUIRED, "Command outcome requires same-key recovery or administrative reconciliation")

    fun inputHash(fields: List<String>): String = HexFormat.of().formatHex(
        MessageDigest.getInstance("SHA-256").digest(mapper.writeValueAsBytes(fields)))

    private fun ensureIdentity(receipt: AdminActionCommand, actorId: UUID, operation: AdminActionOperation,
        targetId: UUID?, hash: String) {
        if (receipt.actorId != actorId || receipt.operation != operation || receipt.targetId != targetId || receipt.inputHash != hash)
            throw ApiException(HttpStatus.CONFLICT, ErrorCodes.CONFLICT, "Command key is already bound to another input")
    }

    private fun replay(receipt: AdminActionCommand, actorId: UUID, operation: AdminActionOperation,
        targetId: UUID?, hash: String): AdminActionResult {
        ensureIdentity(receipt, actorId, operation, targetId, hash)
        when (receipt.state) {
            AdminActionCommandState.PENDING -> throw ApiException(HttpStatus.CONFLICT,
                ErrorCodes.IDEMPOTENT_REPLAY_IN_PROGRESS, "Command is still in progress")
            AdminActionCommandState.UNKNOWN -> throw ApiException(HttpStatus.CONFLICT,
                ErrorCodes.ADMIN_COMMAND_RECOVERY_REQUIRED, "Command requires recovery before it can be retried")
            AdminActionCommandState.FAILED -> throw ApiException(HttpStatus.CONFLICT,
                ErrorCodes.CONFLICT, "Command did not succeed; this key cannot execute another action")
            AdminActionCommandState.SUCCEEDED -> return AdminActionResult(receipt.resultResourceId,
                receipt.resultDecisionId, receipt.resultMessageId)
        }
    }
}
