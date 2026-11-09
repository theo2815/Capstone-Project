package com.quickpitik.service.removal

import com.quickpitik.config.PhotoRemovalProperties
import com.quickpitik.config.RekognitionProperties
import com.quickpitik.entity.RunnerLivenessBand
import com.quickpitik.entity.RunnerRiskFlag
import org.junit.jupiter.api.Test
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

class RunnerAutoHidePolicyTest {

    private val owner = UUID.randomUUID()
    private val otherOwner = UUID.randomUUID()

    private fun policy(
        maxPerRequest: Int = 30,
        dailyPerPhotographer: Int = 50,
        minSimilarity: Double = 0.99,
        floor: Double = 0.8,
    ) = RunnerAutoHidePolicy(
        PhotoRemovalProperties(
            runner = PhotoRemovalProperties.Runner(
                autoHideMinSimilarity = minSimilarity,
                autoHideMaxPerRequest = maxPerRequest,
                photographerDailyAutoHides = dailyPerPhotographer,
            ),
        ),
        RekognitionProperties(faceMatchThreshold = floor),
    )

    private fun photo(similarity: Double = 0.995, ownerId: UUID? = owner, faceCount: Int? = 1) =
        CandidatePhoto(UUID.randomUUID(), ownerId, similarity, faceCount)

    private fun input(
        candidates: List<CandidatePhoto>,
        band: RunnerLivenessBand = RunnerLivenessBand.PASS,
        emailVerified: Boolean = true,
        priorRejection: Boolean = false,
        possiblyTruncated: Boolean = false,
        usedToday: Map<UUID, Long> = emptyMap(),
    ) = AutoHideInput(band, emailVerified, priorRejection, possiblyTruncated, candidates, usedToday)

    @Test fun `a clean pass hides the auto band and suggests the rest`() {
        val high = photo(0.995)
        val low = photo(0.9)
        val result = policy().decide(input(listOf(high, low)))
        assertEquals(listOf(high), result.autoHide)
        assertEquals(listOf(low), result.suggest)
        assertTrue(result.flags.isEmpty())
    }

    @Test fun `0_9899 suggests and 0_99 hides`() {
        val below = photo(0.9899)
        val at = photo(0.99)
        val result = policy().decide(input(listOf(below, at)))
        assertEquals(listOf(at), result.autoHide)
        assertEquals(listOf(below), result.suggest)
    }

    @Test fun `an uncertain liveness band holds everything`() = assertHeld(
        input(listOf(photo()), band = RunnerLivenessBand.UNCERTAIN), RunnerRiskFlag.LIVENESS_UNCERTAIN)

    @Test fun `an unverified email holds everything`() = assertHeld(
        input(listOf(photo()), emailVerified = false), RunnerRiskFlag.UNVERIFIED_EMAIL)

    @Test fun `a prior rejection holds everything`() = assertHeld(
        input(listOf(photo()), priorRejection = true), RunnerRiskFlag.PRIOR_REJECTION)

    @Test fun `a possibly truncated result holds everything`() = assertHeld(
        input(listOf(photo()), possiblyTruncated = true), RunnerRiskFlag.POSSIBLY_TRUNCATED)

    private fun assertHeld(input: AutoHideInput, flag: RunnerRiskFlag) {
        val result = policy().decide(input)
        assertTrue(result.autoHide.isEmpty())
        assertEquals(input.candidates, result.suggest)
        assertEquals(setOf(flag), result.flags)
    }

    @Test fun `31 auto matches hold all 31 with OVER_REQUEST_CAP`() {
        val photos = List(31) { photo() }
        val result = policy().decide(input(photos))
        assertTrue(result.autoHide.isEmpty())
        assertEquals(31, result.suggest.size)
        assertEquals(setOf(RunnerRiskFlag.OVER_REQUEST_CAP), result.flags)
    }

    @Test fun `30 auto matches hide 30`() {
        val result = policy().decide(input(List(30) { photo() }))
        assertEquals(30, result.autoHide.size)
        assertTrue(result.flags.isEmpty())
    }

    @Test fun `max-per-request 0 hides nothing and flags AUTO_HIDE_OFF only, not OVER_REQUEST_CAP`() {
        val result = policy(maxPerRequest = 0).decide(input(listOf(photo())))
        assertTrue(result.autoHide.isEmpty())
        assertEquals(1, result.suggest.size)
        assertEquals(setOf(RunnerRiskFlag.AUTO_HIDE_OFF), result.flags)
    }

    @Test fun `45 used plus 6 new for one owner holds that owner's 6 only`() {
        val mine = List(6) { photo(ownerId = owner) }
        val theirs = photo(ownerId = otherOwner)
        val result = policy().decide(input(mine + theirs, usedToday = mapOf(owner to 45L)))
        assertEquals(listOf(theirs), result.autoHide)
        assertEquals(mine, result.suggest)
        assertEquals(setOf(RunnerRiskFlag.PHOTOGRAPHER_BUDGET), result.flags)
    }

    @Test fun `exactly filling the daily budget still hides`() {
        val result = policy().decide(input(List(5) { photo() }, usedToday = mapOf(owner to 45L)))
        assertEquals(5, result.autoHide.size)
        assertTrue(result.flags.isEmpty())
    }

    @Test fun `a candidate with no owner is suggested`() {
        val orphan = photo(ownerId = null)
        val result = policy().decide(input(listOf(orphan)))
        assertTrue(result.autoHide.isEmpty())
        assertEquals(listOf(orphan), result.suggest)
    }

    @Test fun `a group or unknown-face-count photo stays in the auto band and flags GROUP_PHOTOS`() {
        val group = photo(faceCount = 3)
        val unknown = photo(faceCount = null)
        val single = photo(faceCount = 1)
        val result = policy().decide(input(listOf(group, unknown, single)))
        assertEquals(listOf(group, unknown, single), result.autoHide)
        assertEquals(setOf(RunnerRiskFlag.GROUP_PHOTOS), result.flags)
    }

    @Test fun `single-face photos raise no group flag`() {
        assertTrue(policy().decide(input(listOf(photo(faceCount = 1)))).flags.isEmpty())
    }

    @Test fun `liveness 89_9 is UNCERTAIN, 90 passes, 49_9 is a fail, EXPIRED is a fail`() {
        val p = policy()
        assertEquals(RunnerLivenessBand.PASS, p.livenessBand(90.0, "SUCCEEDED"))
        assertEquals(RunnerLivenessBand.UNCERTAIN, p.livenessBand(89.9, "SUCCEEDED"))
        assertEquals(RunnerLivenessBand.UNCERTAIN, p.livenessBand(50.0, "SUCCEEDED"))
        assertNull(p.livenessBand(49.9, "SUCCEEDED"))
        assertNull(p.livenessBand(99.0, "EXPIRED"))
        assertNull(p.livenessBand(99.0, "FAILED"))
        assertNull(p.livenessBand(99.0, "IN_PROGRESS"))
        assertNull(p.livenessBand(null, "SUCCEEDED"))
    }

    @Test fun `an auto-hide value below the provider floor fails at construction`() {
        assertFailsWith<IllegalArgumentException> { policy(minSimilarity = 0.7, floor = 0.8) }
    }
}
