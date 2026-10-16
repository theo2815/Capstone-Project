package com.quickpitik.dto.photos

import com.quickpitik.entity.RemovalActorType
import java.util.UUID

enum class PhotoErasurePurpose { PHOTOGRAPHER_DELETE, PRIVACY_REMOVAL }
enum class PhotoErasureOutcome { PENDING, RETRY_REQUIRED, COMPLETE }

data class PhotoErasureCommand(
    val photoId: UUID,
    val purpose: PhotoErasurePurpose,
    val authorizedActorType: RemovalActorType,
    val authorizedActorId: UUID,
    val reason: String,
    val requestId: UUID? = null,
    val itemId: UUID? = null,
    val commandId: UUID,
)
