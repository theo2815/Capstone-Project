package com.quickpitik.service.removal

import com.quickpitik.common.OffsetLimitPageable
import com.quickpitik.dto.removal.*
import com.quickpitik.entity.*
import com.quickpitik.repository.*
import org.springframework.data.domain.Pageable
import org.springframework.data.domain.Sort
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

/** A runner's own deletion requests (RD-5): status, counts and a timeline, never photos or anyone else's data. */
@Service
@Transactional(readOnly = true)
class RunnerPhotoRemovalQueryService(
    private val requests: PhotoRemovalRequestRepository,
    private val items: PhotoRemovalItemRepository,
    private val history: PhotoRemovalHistoryRepository,
) {
    fun list(runnerId: UUID, offset: Int?, limit: Int?): RemovalPage<RunnerRemovalRequestDto> {
        val params = removalPagination(offset, limit)
        val page = requests.findByRequesterUserIdAndSource(runnerId, RemovalRequestSource.RUNNER,
            OffsetLimitPageable(params, Sort.by(Sort.Direction.DESC, "createdAt").and(Sort.by("id"))))
        val counts = if (page.isEmpty) emptyMap()
            else items.countStatesByRequestIds(page.content.map { it.id }).groupBy { it.requestId }
        return RemovalPage(page.content.map { summary(RunnerStatusView(it, counts[it.id].orEmpty())) },
            page.totalElements, params.offset, params.limit)
    }

    fun detail(runnerId: UUID, requestId: UUID): RunnerRemovalRequestDetailDto =
        detailOf(requests.findByIdAndRequesterUserIdAndSource(requestId, runnerId, RemovalRequestSource.RUNNER) ?: removalNotFound())

    fun view(request: PhotoRemovalRequest) = RunnerStatusView(request, items.countStatesByRequestIds(listOf(request.id)))

    /** The caller has already scoped [request] to its runner. */
    internal fun detailOf(request: PhotoRemovalRequest): RunnerRemovalRequestDetailDto {
        val s = summary(view(request))
        // R3: history is written per item, so one step per (step, command), at its first row's time.
        val timeline = history.findByRequestIdOrderByOccurredAtAscIdAsc(request.id, Pageable.unpaged())
            .mapNotNull { row -> timelineStep(row)?.let { Triple(it, row.commandId, row.occurredAt) } }
            .distinctBy { it.first to it.second }
            .map { RunnerTimelineStepDto(it.third, it.first) }
        return RunnerRemovalRequestDetailDto(s.id, s.eventId, s.eventName, s.status, s.createdAt, s.updatedAt, s.version,
            s.canWithdraw, s.counts, timeline)
    }

    private fun summary(view: RunnerStatusView): RunnerRemovalRequestDto {
        val r = view.request
        fun n(vararg states: RemovalItemState) = view.counts.filter { it.state in states }.sumOf { it.count }.toInt()
        return RunnerRemovalRequestDto(r.id, r.eventIdSnapshot, r.eventNameSnapshot, RunnerRemovalStatus.of(view).name,
            r.createdAt, r.updatedAt, r.version, removalClosable(r.status, view.counts.map { it.state }),
            RunnerRemovalCountsDto(
                hidden = n(RemovalItemState.VERIFIED, RemovalItemState.RETAINED_RESTRICTED, RemovalItemState.REMOVAL_AUTHORIZED),
                suggested = n(RemovalItemState.CANDIDATE), removed = n(RemovalItemState.REMOVED), kept = n(RemovalItemState.RETAINED)))
    }
}

// A fixed allow-list: every other row (notices, evidence, erasure, admin notes) never reaches the runner.
internal fun timelineStep(row: PhotoRemovalHistory): String? = when (row.action) {
    RemovalHistoryActions.CREATED -> "SUBMITTED"
    RemovalHistoryActions.VERIFIED -> when (row.actorType) {
        RemovalActorType.SYSTEM -> "PHOTOS_HIDDEN"
        RemovalActorType.ADMIN -> "SENT_FOR_REVIEW"
        else -> null
    }
    RemovalHistoryActions.ADMIN_DECIDED -> when (row.metadata["decision"]) {
        RemovalDecision.MORE_EVIDENCE.name -> "MORE_INFO_NEEDED"
        RemovalDecision.REMOVE.name -> "APPROVED"
        RemovalDecision.RETAIN.name -> "KEPT"
        else -> null
    }
    RemovalHistoryActions.CANCEL_REQUESTED -> if (row.actorType == RemovalActorType.RUNNER) "WITHDRAWN" else null
    RemovalHistoryActions.RESOLVED -> "CLOSED"
    else -> null
}
