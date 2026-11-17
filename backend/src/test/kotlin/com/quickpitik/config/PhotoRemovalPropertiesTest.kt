package com.quickpitik.config

import org.junit.jupiter.api.Test
import java.time.Duration
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PhotoRemovalPropertiesTest {

    @Test fun `runner defaults match the settled starting values`() {
        val runner = PhotoRemovalProperties().runner
        assertEquals(90.0, runner.livenessPassMin)
        assertEquals(50.0, runner.livenessUncertainMin)
        assertEquals(0.99, runner.autoHideMinSimilarity)
        assertEquals(30, runner.autoHideMaxPerRequest)
        assertEquals(50, runner.photographerDailyAutoHides)
        assertEquals(50, runner.candidateLimit)
        assertEquals(Duration.ofMinutes(10), runner.emailClaimStaleAfter)
    }

    @Test fun `similarity above 1_0 or uncertain above pass fails at startup`() {
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(autoHideMinSimilarity = 1.01) }
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(livenessUncertainMin = 91.0, livenessPassMin = 90.0) }
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(livenessPassMin = 101.0) }
    }

    @Test fun `caps and the candidate limit reject out-of-range values`() {
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(autoHideMaxPerRequest = -1) }
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(photographerDailyAutoHides = -1) }
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(candidateLimit = 0) }
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(candidateLimit = 4097) }
    }

    @Test fun `email-claim-stale-after defaults to 10 minutes and zero fails at startup`() {
        assertFailsWith<IllegalArgumentException> { PhotoRemovalProperties.Runner(emailClaimStaleAfter = Duration.ZERO) }
    }

    @Test fun `the real application yml binds into the nested runner block with its defaults`() {
        val env = org.springframework.core.env.StandardEnvironment()
        org.springframework.boot.env.YamlPropertySourceLoader().load("yml", org.springframework.core.io.ClassPathResource("application.yml"))
            .forEach { env.propertySources.addLast(it) }
        val bound = org.springframework.boot.context.properties.bind.Binder.get(env)
            .bind("app.photo-removal", PhotoRemovalProperties::class.java).get()
        assertEquals(PhotoRemovalProperties().runner, bound.runner)
    }

    @Test fun `application yml wires every runner key to an env override`() {
        val yml = javaClass.getResource("/application.yml")!!.readText()
        listOf(
            "RUNNER_REMOVAL_LIVENESS_PASS:90", "RUNNER_REMOVAL_LIVENESS_UNCERTAIN:50",
            "RUNNER_REMOVAL_AUTO_HIDE_SIMILARITY:0.99", "RUNNER_REMOVAL_AUTO_HIDE_MAX:30",
            "RUNNER_REMOVAL_PHOTOGRAPHER_DAILY:50", "RUNNER_REMOVAL_CANDIDATE_LIMIT:50",
            "RUNNER_REMOVAL_EMAIL_CLAIM_STALE_AFTER:PT10M",
        ).forEach { assertTrue(it in yml, "missing $it in application.yml") }
    }
}
