package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceMode
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.service.events.postgresTimestamp
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Component
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.util.UUID

// RV-1.2 (plan 5.5): counts a photo's words into its batch once, after the indexing outcome has committed.
// Never throws: a failure only leaves this photo's words out ("words from N of M photos").
@Component
class RelevanceWordWriter(
    private val repository: BatchRelevanceRepository,
    private val switch: RelevanceSwitch,
    private val transactions: TransactionTemplate,
    private val clock: Clock,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    fun record(photoId: UUID, eventId: UUID, photographerId: UUID, words: List<String>) {
        if (switch.mode == RelevanceMode.OFF) return
        val clean = cleanWords(words)
        if (clean.isEmpty()) return
        try {
            transactions.execute {
                repository.ensureBatch(eventId, photographerId)
                // No row: a deleted event or a suspended photographer.
                val batch = repository.findByEventIdAndPhotographerId(eventId, photographerId) ?: return@execute
                if (repository.markWordsCounted(photoId, batch.id, clock.postgresTimestamp()) == 1) {
                    repository.addWords(batch.id, clean.sorted().toTypedArray())
                }
            }
        } catch (ex: Exception) {
            // Ids only: the words and the SQL text stay out of the log.
            log.warn("Relevance words not recorded photo={} event={} ({})", photoId, eventId, ex.javaClass.simpleName)
        }
    }
}
