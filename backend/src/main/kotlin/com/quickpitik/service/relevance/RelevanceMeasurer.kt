package com.quickpitik.service.relevance

import com.quickpitik.config.RelevanceProperties
import com.quickpitik.repository.BatchRelevanceRepository
import com.quickpitik.repository.PhotoToMeasure
import com.quickpitik.service.ai.AiApiException
import com.quickpitik.service.ai.ImageCheck
import com.quickpitik.service.ai.RekognitionAiClient
import com.quickpitik.service.events.postgresTimestamp
import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.MeterRegistry
import io.micrometer.core.instrument.Timer
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.ObjectProvider
import org.springframework.beans.factory.annotation.Qualifier
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import org.springframework.transaction.support.TransactionTemplate
import java.time.Clock
import java.util.UUID
import java.util.concurrent.CompletableFuture
import java.util.concurrent.Executor
import java.util.concurrent.RejectedExecutionException

// RV-1.6 (plan 5.6, 5.8): measures sampled photos, each on the relevanceScan pool: read the original from
// storage, one DetectLabels call, then one short write while the caller still holds the batch's claim. No
// transaction is open around the read or the AWS call (backend rule 4). Raw scores are stored; junk is decided
// at read time. Part 2's full scan reuses this. The caller checks RelevanceSwitch.canMeasure first.
@Service
class RelevanceMeasurer(
    private val batches: BatchRelevanceRepository,
    private val storage: StorageService,
    private val rekognition: ObjectProvider<RekognitionAiClient>,
    @Qualifier("relevanceScan") private val pool: Executor,
    private val props: RelevanceProperties,
    private val clock: Clock,
    private val transactions: TransactionTemplate,
    private val registry: MeterRegistry,
) {
    private val log = LoggerFactory.getLogger(javaClass)

    // Waits for every photo and returns how many measurements were written.
    fun measure(batchId: UUID, token: UUID, photos: List<PhotoToMeasure>): Int {
        val client = rekognition.getIfAvailable() ?: return 0
        val runs = photos.map { photo ->
            try {
                CompletableFuture.supplyAsync({ measureOne(client, batchId, token, photo) }, pool)
            } catch (ex: RejectedExecutionException) {
                CompletableFuture.completedFuture(0) // pool full: the photo is still to measure next tick
            }
        }
        return runs.sumOf { run ->
            try {
                run.join()
            } catch (ex: Exception) {
                log.warn("Relevance measure write failed batch={} (retried next tick): {}", batchId, ex.cause?.message ?: ex.message)
                0
            }
        }
    }

    private fun measureOne(client: RekognitionAiClient, batchId: UUID, token: UUID, photo: PhotoToMeasure): Int {
        val timer = Timer.start(registry)
        try {
            val check = try {
                client.imageCheck(storage.getBytes(photo.s3Key))
            } catch (ex: AiApiException) {
                // A 422 is a bad image: never junk (G4). A throttle, a 5xx or a timeout may pass next tick.
                return if (ex.status == HttpStatus.UNPROCESSABLE_ENTITY) error(batchId, token, photo, "BAD_IMAGE") else retry(batchId, token, photo)
            } catch (ex: Exception) {
                return if (missing(photo.s3Key)) error(batchId, token, photo, "NO_FILE") else retry(batchId, token, photo)
            }
            val now = clock.postgresTimestamp()
            val written = transactions.execute {
                batches.saveMeasurement(photo.photoId, batchId, token, now, check.fgSharpness, check.fgBrightness, check.sharpness,
                    check.brightness, confidence(check, setOf(PERSON)), confidence(check, props.junk.raceLabels.toSet()))
            } ?: 0
            count("ok")
            return written
        } finally {
            timer.stop(registry.timer("qp.relevance.measure"))
        }
    }

    // An absent label is stored as 0, never null (plan 5.6).
    private fun confidence(check: ImageCheck, names: Set<String>): Double =
        check.labels.filter { it.name in names }.maxOfOrNull { it.confidence } ?: 0.0

    // A storage error that is not a missing file (an R2 outage) is tried again.
    private fun missing(key: String): Boolean = try {
        !storage.exists(key)
    } catch (ex: Exception) {
        false
    }

    private fun error(batchId: UUID, token: UUID, photo: PhotoToMeasure, error: String): Int {
        log.info("Relevance measure error photo={} error={}", photo.photoId, error)
        transactions.execute { batches.saveMeasureError(photo.photoId, batchId, token, error) }
        count("error")
        return 0
    }

    private fun retry(batchId: UUID, token: UUID, photo: PhotoToMeasure): Int {
        transactions.execute { batches.saveMeasureRetry(photo.photoId, batchId, token, props.check.measureMaxTries) }
        count("retry")
        return 0
    }

    private fun count(result: String) = registry.counter("qp.relevance.measured", "result", result).increment()

    private companion object {
        const val PERSON = "Person"
    }
}
