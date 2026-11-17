package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RekognitionProperties
import com.quickpitik.entity.RunnerLivenessBand
import com.quickpitik.entity.RunnerRiskFlag
import org.springframework.stereotype.Component
import java.util.UUID

data class CandidatePhoto(val photoId: UUID, val ownerId: UUID?, val similarity: Double, val faceCount: Int?)

data class AutoHideInput(
    val livenessBand: RunnerLivenessBand,
    val emailVerified: Boolean,
    val priorRejection: Boolean,
    val possiblyTruncated: Boolean,
    val candidates: List<CandidatePhoto>,
    val ownerAutoHidesToday: Map<UUID, Long>,
)

data class AutoHideDecision(val autoHide: List<CandidatePhoto>, val suggest: List<CandidatePhoto>, val flags: Set<RunnerRiskFlag>)

// Decides which of a runner's matches hide at once and which only go to an admin as suggestions.
@Component
class RunnerAutoHidePolicy(props: PhotoRemovalProperties, rekognition: RekognitionProperties) {
    private val config = props.runner

    init {
        require(config.autoHideMinSimilarity >= rekognition.faceMatchThreshold) {
            "auto-hide similarity must be at least the provider match floor"
        }
    }

    /** Null means a failed check (nothing is stored); an unfinished session never scores. */
    fun livenessBand(confidence: Double?, status: String): RunnerLivenessBand? = when {
        status != "SUCCEEDED" || confidence == null -> null
        confidence >= config.livenessPassMin -> RunnerLivenessBand.PASS
        confidence >= config.livenessUncertainMin -> RunnerLivenessBand.UNCERTAIN
        else -> null
    }

    fun decide(input: AutoHideInput): AutoHideDecision {
        val flags = mutableSetOf<RunnerRiskFlag>()
        if (input.livenessBand != RunnerLivenessBand.PASS) flags += RunnerRiskFlag.LIVENESS_UNCERTAIN
        if (!input.emailVerified) flags += RunnerRiskFlag.UNVERIFIED_EMAIL
        if (input.priorRejection) flags += RunnerRiskFlag.PRIOR_REJECTION
        if (input.possiblyTruncated) flags += RunnerRiskFlag.POSSIBLY_TRUNCATED
        if (config.autoHideMaxPerRequest == 0) flags += RunnerRiskFlag.AUTO_HIDE_OFF

        val band = input.candidates.filter { it.similarity >= config.autoHideMinSimilarity && it.ownerId != null }
        if (config.autoHideMaxPerRequest > 0 && band.size > config.autoHideMaxPerRequest) flags += RunnerRiskFlag.OVER_REQUEST_CAP

        var auto = if (flags.isEmpty()) band else emptyList()
        val overBudget = auto.groupBy { it.ownerId }
            .filter { (owner, items) -> (input.ownerAutoHidesToday[owner] ?: 0L) + items.size > config.photographerDailyAutoHides }
            .keys
        if (overBudget.isNotEmpty()) {
            flags += RunnerRiskFlag.PHOTOGRAPHER_BUDGET
            auto = auto.filter { it.ownerId !in overBudget }
        }
        if (auto.any { it.faceCount == null || it.faceCount >= 2 }) flags += RunnerRiskFlag.GROUP_PHOTOS

        val hidden = auto.toSet()
        return AutoHideDecision(auto, input.candidates.filter { it !in hidden }, flags)
    }
}
