package com.quickpitik.service.earnings

import com.quickpitik.entity.PayoutCycleStatus
import com.quickpitik.entity.PayoutStatusHistory
import com.quickpitik.repository.PayoutStatusHistoryRepository
import org.springframework.stereotype.Component
import java.util.UUID

// Single writer for the payout lifecycle audit trail. Both the photographer
// (request/cancel) and admin (approve/hold/reject/mark-*) transition paths call
// this so every status change lands in payout_status_history in the same TX as
// the change itself.
@Component
class PayoutStatusRecorder(
    private val payoutStatusHistoryRepository: PayoutStatusHistoryRepository,
) {
    fun record(
        cycleId: String,
        from: PayoutCycleStatus?,
        to: PayoutCycleStatus,
        actorType: String,
        actorId: UUID?,
        reason: String? = null,
    ) {
        payoutStatusHistoryRepository.save(
            PayoutStatusHistory(
                payoutCycleId = cycleId,
                fromStatus = from?.wire,
                toStatus = to.wire,
                actorType = actorType,
                actorId = actorId,
                reason = reason?.take(255),
            ),
        )
    }
}
