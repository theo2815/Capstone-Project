package com.quickpitik.support

import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.jdbc.core.JdbcTemplate
import kotlin.test.assertEquals

// AF-1: the one migration of the AA review fixes (R-7, R-8, R-10) adds three columns with defaults.
class AaReviewFixesSchemaIntegrationTest : PostgresIntegrationTest() {

    @Autowired private lateinit var jdbc: JdbcTemplate

    @Test
    fun `the three review-fix columns exist with their defaults`() {
        val columns = jdbc.queryForList(
            "SELECT table_name || '.' || column_name || ' ' || data_type || ' ' || is_nullable || ' ' || " +
                "coalesce(column_default, '-') FROM information_schema.columns WHERE table_schema = current_schema() " +
                "AND (table_name, column_name) IN (('photographer_seller_details', 'failed_auto_checks'), " +
                "('photographer_settings', 'verification_admin_hold'), ('users', 'identity_notice_sent_for')) ORDER BY 1",
            String::class.java,
        )
        assertEquals(
            listOf(
                "photographer_seller_details.failed_auto_checks integer NO 0",
                "photographer_settings.verification_admin_hold boolean NO false",
                "users.identity_notice_sent_for date YES -",
            ),
            columns,
        )
    }
}
