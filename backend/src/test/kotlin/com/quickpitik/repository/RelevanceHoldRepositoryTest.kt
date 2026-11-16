package com.quickpitik.repository

import org.junit.jupiter.api.Test
import org.springframework.data.jpa.repository.Query
import kotlin.test.assertTrue

class RelevanceHoldRepositoryTest {

    // RVF-11 (RVR-14): the chunk restores lock their hold rows in photo order, as the hides and restoreByPhotographer do,
    // so a restore and a hide of the same batch wait for each other instead of deadlocking.
    @Test
    fun `the three chunk restores lock their rows in photo order`() {
        for (name in listOf("restoreNotJunkChunk", "restoreChunk", "restoreByAdminChunk")) {
            val sql = RelevanceHoldRepository::class.java.methods.single { it.name == name }.getAnnotation(Query::class.java).value
            val subselect = sql.substringAfter("WHERE id IN (").substringBefore("RETURNING")
            assertTrue(Regex("""ORDER BY h?\.?photo_id\b.*FOR UPDATE""", RegexOption.DOT_MATCHES_ALL).containsMatchIn(subselect),
                "$name must lock its rows in photo order: $subselect")
        }
    }
}
