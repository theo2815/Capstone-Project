package com.quickpitik.service.ai

import com.quickpitik.config.RekognitionProperties
import com.quickpitik.dto.ai.BibDetection
import com.quickpitik.dto.ai.BibsRecognizeResult
import com.quickpitik.dto.ai.FaceBBox
import com.quickpitik.dto.ai.FaceDetection
import com.quickpitik.dto.ai.FaceMatch
import com.quickpitik.dto.ai.FacesDetectResult
import com.quickpitik.dto.ai.FacesEnrollResult
import com.quickpitik.dto.ai.FacesSearchResult
import com.quickpitik.service.image.ExifOrientation
import org.slf4j.LoggerFactory
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.context.annotation.Primary
import org.springframework.http.HttpStatus
import org.springframework.stereotype.Service
import software.amazon.awssdk.core.SdkBytes
import software.amazon.awssdk.services.rekognition.RekognitionClient
import software.amazon.awssdk.services.rekognition.model.CreateCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DeleteCollectionRequest
import software.amazon.awssdk.services.rekognition.model.DetectFacesRequest
import software.amazon.awssdk.services.rekognition.model.DetectTextRequest
import software.amazon.awssdk.services.rekognition.model.Image
import software.amazon.awssdk.services.rekognition.model.IndexFacesRequest
import software.amazon.awssdk.services.rekognition.model.InvalidParameterException
import software.amazon.awssdk.services.rekognition.model.QualityFilter
import software.amazon.awssdk.services.rekognition.model.RekognitionException
import software.amazon.awssdk.services.rekognition.model.ResourceAlreadyExistsException
import software.amazon.awssdk.services.rekognition.model.ResourceNotFoundException
import software.amazon.awssdk.services.rekognition.model.SearchFacesByImageRequest
import software.amazon.awssdk.services.rekognition.model.TextTypes
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import javax.imageio.ImageIO
import kotlin.math.roundToInt

// AWS Rekognition implementation of FaceBibProvider — active when
// app.ai.provider=rekognition. Faces use one collection per event with
// IndexFaces / SearchFacesByImage keyed by ExternalImageId="{eventId}.{photoId}"
// (so every face in a photo maps back to that photo, and multi-runner photos
// match on any face). Bibs use DetectText. Cloudflare R2 holds the bytes;
// Rekognition can't read R2, so callers pass image bytes inline — over the 5 MB
// cap they get downscaled first.
@Service
@Primary
@ConditionalOnProperty(prefix = "app.ai", name = ["provider"], havingValue = "rekognition")
class RekognitionAiClient(
    private val rekognition: RekognitionClient,
    private val props: RekognitionProperties,
) : FaceBibProvider {
    private val log = LoggerFactory.getLogger(javaClass)

    // Collections we've ensured exist this JVM — skips a redundant create on the
    // hot enroll path. Correctness never depends on it: indexFaces re-creates on
    // ResourceNotFound, so a stale/missing entry self-heals.
    private val knownCollections = ConcurrentHashMap.newKeySet<String>()

    override fun facesDetect(file: ByteArray, contentType: String, filename: String): FacesDetectResult =
        guarded("detectFaces") {
            val resp = rekognition.detectFaces(DetectFacesRequest.builder().image(imageOf(file)).build())
            FacesDetectResult(
                faces = resp.faceDetails().map { fd ->
                    val conf = (fd.confidence() ?: 0f).toDouble() / 100.0
                    FaceDetection(confidence = conf, bbox = fd.boundingBox()?.let { bb ->
                        FaceBBox(
                            x1 = (bb.left() ?: 0f).toDouble(),
                            y1 = (bb.top() ?: 0f).toDouble(),
                            x2 = ((bb.left() ?: 0f) + (bb.width() ?: 0f)).toDouble(),
                            y2 = ((bb.top() ?: 0f) + (bb.height() ?: 0f)).toDouble(),
                            confidence = conf,
                        )
                    })
                },
            )
        }

    override fun facesEnroll(
        file: ByteArray,
        contentType: String,
        filename: String,
        personName: String,
        personId: String?,
        eventId: UUID,
    ): FacesEnrollResult = guarded("indexFaces") {
        val collectionId = collectionFor(eventId)
        // personName is the photo id (set by PhotoIndexingService); the composite
        // ExternalImageId lets deleteFacesByEvent locate the collection and search
        // map matches straight back to the photo.
        val externalId = "$eventId.$personName"
        val resp = indexWithCollection(collectionId, externalId, file)
        val count = resp.faceRecords().size
        if (count == 0) {
            // Mirror ai-api: no usable face is a benign outcome, not a failure.
            throw AiApiException(HttpStatus.UNPROCESSABLE_ENTITY, "NO_FACES", "No faces detected in image")
        }
        FacesEnrollResult(person_id = externalId, faces_enrolled = count)
    }

    override fun facesSearch(
        file: ByteArray,
        contentType: String,
        filename: String,
        eventId: UUID,
        threshold: Double,
        topK: Int,
    ): FacesSearchResult = guarded("searchFacesByImage") {
        val collectionId = collectionFor(eventId)
        // Never search looser than the configured floor — face search must not
        // surface someone else's photos. The backend's server-side re-filter is
        // an additional (weaker) net on top of this.
        val effective = maxOf(threshold, props.faceMatchThreshold)
        val resp = try {
            rekognition.searchFacesByImage(
                SearchFacesByImageRequest.builder()
                    .collectionId(collectionId)
                    .image(imageOf(file))
                    .faceMatchThreshold((effective * 100).toFloat())
                    .maxFaces(topK.coerceIn(1, 4096))
                    .build(),
            )
        } catch (ex: ResourceNotFoundException) {
            // No collection yet → nothing enrolled for this event.
            return@guarded FacesSearchResult(matches = emptyList())
        } catch (ex: InvalidParameterException) {
            // Rekognition raises this when it finds no face in the query image.
            return@guarded FacesSearchResult(matches = emptyList())
        }
        val matches = resp.faceMatches()
            .filter { !it.face().externalImageId().isNullOrBlank() }
            .groupBy { it.face().externalImageId()!! }
            .map { (extId, group) ->
                FaceMatch(person_id = extId, similarity = (group.maxOf { it.similarity() ?: 0f }).toDouble() / 100.0)
            }
        FacesSearchResult(matches = matches)
    }

    override fun bibsRecognize(
        file: ByteArray,
        contentType: String,
        filename: String,
        minChars: Int?,
    ): BibsRecognizeResult = guarded("detectText") {
        val min = minChars ?: props.bibMinChars
        val resp = rekognition.detectText(DetectTextRequest.builder().image(imageOf(file)).build())
        val detections = resp.textDetections()
            // WORD granularity: a bib is a single token. LINE would merge a bib
            // with neighbouring sponsor text.
            .filter { it.type() == TextTypes.WORD }
            .mapNotNull { td ->
                val raw = td.detectedText()?.trim().orEmpty()
                val digits = raw.count(Char::isDigit)
                if (raw.isNotEmpty() && digits >= min && raw.all { it.isLetterOrDigit() || it == '-' }) {
                    BibDetection(bib_number = raw.uppercase(), confidence = (td.confidence() ?: 0f).toDouble() / 100.0)
                } else {
                    null
                }
            }
        BibsRecognizeResult(detections = detections)
    }

    override fun deleteFacesPerson(personId: String) {
        // ponytail: basic collections can't locate faces by ExternalImageId, so
        // one photo's faces can't be surgically deleted. Harmless here — re-index
        // leaves duplicate faces under the same ExternalImageId (same photo → same
        // match) and event deletion drops the whole collection. Upgrade path:
        // Rekognition Users (CreateUser/AssociateFaces) if per-photo erasure is needed.
        log.debug("Rekognition provider: per-person delete is a no-op (personId={})", personId)
    }

    override fun deleteFacesByEvent(eventId: UUID) {
        val collectionId = collectionFor(eventId)
        try {
            rekognition.deleteCollection(DeleteCollectionRequest.builder().collectionId(collectionId).build())
            knownCollections.remove(collectionId)
            log.info("Deleted Rekognition collection {}", collectionId)
        } catch (ex: ResourceNotFoundException) {
            // Never created (no photos enrolled for this event) → nothing to do.
        } catch (ex: RekognitionException) {
            log.warn("Failed to delete Rekognition collection {}: {}", collectionId, ex.awsErrorDetails()?.errorMessage())
        }
    }

    override fun listPersonsForEvent(eventId: UUID): List<AiPersonRef> {
        // The orphan reaper is ai-api-specific; under Rekognition its cleanup
        // happens via event-collection deletion. Empty = safe no-op for the reaper.
        return emptyList()
    }

    private fun indexWithCollection(collectionId: String, externalId: String, file: ByteArray) =
        try {
            ensureCollection(collectionId)
            rekognition.indexFaces(indexRequest(collectionId, externalId, file))
        } catch (ex: ResourceNotFoundException) {
            // Cache said known but the collection was gone (deleted externally, or
            // an ensure race) — create and retry once.
            createCollection(collectionId)
            rekognition.indexFaces(indexRequest(collectionId, externalId, file))
        }

    private fun indexRequest(collectionId: String, externalId: String, file: ByteArray) =
        IndexFacesRequest.builder()
            .collectionId(collectionId)
            .externalImageId(externalId)
            .image(imageOf(file))
            .maxFaces(props.maxFacesPerImage)
            .qualityFilter(QualityFilter.AUTO)
            .build()

    private fun collectionFor(eventId: UUID): String = "${props.collectionPrefix}$eventId"

    private fun ensureCollection(collectionId: String) {
        if (collectionId in knownCollections) return
        createCollection(collectionId)
    }

    private fun createCollection(collectionId: String) {
        try {
            rekognition.createCollection(CreateCollectionRequest.builder().collectionId(collectionId).build())
            log.info("Created Rekognition collection {}", collectionId)
        } catch (ex: ResourceAlreadyExistsException) {
            // Concurrent create, or already present — fine.
        }
        knownCollections.add(collectionId)
    }

    private fun imageOf(file: ByteArray): Image =
        Image.builder().bytes(SdkBytes.fromByteArray(toInlineBytes(file))).build()

    // Rekognition inline images cap at 5 MB. Small images pass through untouched
    // (Rekognition applies EXIF orientation itself). Only oversized ones are
    // decoded, uprighted, downscaled to maxImageDimension, and re-encoded as JPEG.
    private fun toInlineBytes(file: ByteArray): ByteArray {
        if (file.size <= INLINE_SAFE_BYTES) return file
        val source = ImageIO.read(ByteArrayInputStream(file)) ?: return file
        val upright = ExifOrientation.apply(source, ExifOrientation.read(file))
        val longest = maxOf(upright.width, upright.height)
        val ratio = if (longest > props.maxImageDimension) props.maxImageDimension.toDouble() / longest else 1.0
        val w = (upright.width * ratio).roundToInt().coerceAtLeast(1)
        val h = (upright.height * ratio).roundToInt().coerceAtLeast(1)
        val dst = BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
        val g = dst.createGraphics()
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
            g.drawImage(upright, 0, 0, w, h, null)
        } finally {
            g.dispose()
        }
        val out = ByteArrayOutputStream()
        return if (ImageIO.write(dst, "jpeg", out)) out.toByteArray() else file
    }

    private fun <T> guarded(op: String, block: () -> T): T =
        try {
            block()
        } catch (ex: AiApiException) {
            throw ex
        } catch (ex: RekognitionException) {
            log.warn("Rekognition {} failed: {}", op, ex.awsErrorDetails()?.errorMessage() ?: ex.message)
            throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "AWS Rekognition $op failed", ex)
        } catch (ex: Exception) {
            log.warn("Rekognition {} error: {}", op, ex.message)
            throw AiApiException(HttpStatus.SERVICE_UNAVAILABLE, null, "AWS Rekognition $op error", ex)
        }

    private companion object {
        // Leaves margin under Rekognition's 5 MB inline cap before we downscale.
        const val INLINE_SAFE_BYTES = 4_500_000
    }
}
