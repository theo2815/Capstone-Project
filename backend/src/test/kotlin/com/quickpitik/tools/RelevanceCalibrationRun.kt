package com.quickpitik.tools

import com.quickpitik.config.RekognitionProperties
import com.quickpitik.service.ai.ImageCheck
import com.quickpitik.service.ai.RekognitionAiClient
import io.micrometer.core.instrument.simple.SimpleMeterRegistry
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider
import software.amazon.awssdk.regions.Region
import software.amazon.awssdk.services.rekognition.RekognitionClient
import java.io.File

/**
 * RV-0.3 (O7): the owner's one-off calibration run. Skipped unless RELEVANCE_CALIBRATION_DIR is set.
 * Setting it makes real, billed DetectLabels calls (one per JPEG), so only the owner sets it (OS-3).
 *
 * The folder holds hand-sorted sub-folders sharp/, blurry/, night/ and not-race/. Writes
 * build/relevance-calibration/scores.csv and prints, per folder, the 5th/50th/95th percentile of each
 * score and the ten most common labels; the owner reads the two cut-offs and the race labels from that.
 */
class RelevanceCalibrationRun {
    @Test
    @EnabledIfEnvironmentVariable(named = "RELEVANCE_CALIBRATION_DIR", matches = ".+")
    fun `measure the calibration folders`() {
        val root = File(System.getenv("RELEVANCE_CALIBRATION_DIR"))
        require(root.isDirectory) { "RELEVANCE_CALIBRATION_DIR is not a folder: $root" }
        val props = RekognitionProperties()
        val rekognition = RekognitionClient.builder().region(Region.of(props.region))
            .overrideConfiguration { it.apiCallTimeout(props.operationTimeout) }
            .credentialsProvider(DefaultCredentialsProvider.create())
            .build()
        val client = RekognitionAiClient(rekognition, props, SimpleMeterRegistry())

        val out = File("build/relevance-calibration/scores.csv").apply { parentFile.mkdirs() }
        out.bufferedWriter().use { csv ->
            csv.write("folder,file,fg_sharpness,fg_brightness,sharpness,brightness,labels\n")
            for (folder in FOLDERS) {
                val photos = File(root, folder).listFiles { f -> f.isFile && f.extension.lowercase() in JPEG }
                    ?.sortedBy { it.name }.orEmpty()
                val checks = mutableListOf<ImageCheck>()
                for (photo in photos) {
                    // One bad photo must not waste the billed calls already made: note it and go on.
                    val check = runCatching { client.imageCheck(photo.readBytes()) }
                        .onFailure { println("$folder/${photo.name}: not measured (${it.message})") }
                        .getOrNull() ?: continue
                    checks += check
                    val labels = check.labels.joinToString(";") { "${it.name}:${"%.3f".format(it.confidence)}" }
                    csv.write(listOf(folder, photo.name, check.fgSharpness, check.fgBrightness, check.sharpness, check.brightness, labels)
                        .joinToString(",") { quote(it?.toString() ?: "") } + "\n")
                }
                println(summary(folder, photos.size, checks))
            }
        }
        rekognition.close()
        println("Wrote ${out.absolutePath}")
    }

    private fun summary(folder: String, files: Int, checks: List<ImageCheck>): String {
        fun line(name: String, values: List<Double?>): String {
            val v = values.filterNotNull()
            val p = listOf(5.0, 50.0, 95.0).joinToString(" / ") { pct -> percentile(v, pct)?.let { "%.1f".format(it) } ?: "-" }
            return "  $name (n=${v.size}): p5 / p50 / p95 = $p"
        }
        val top = checks.flatMap { c -> c.labels.map { it.name } }.groupingBy { it }.eachCount()
            .entries.sortedByDescending { it.value }.take(10).joinToString(", ") { "${it.key} (${it.value})" }
        return listOf(
            "$folder: $files files, ${checks.size} measured",
            line("fg_sharpness", checks.map { it.fgSharpness }),
            line("fg_brightness", checks.map { it.fgBrightness }),
            line("sharpness", checks.map { it.sharpness }),
            line("brightness", checks.map { it.brightness }),
            "  top labels: $top",
        ).joinToString("\n")
    }

    private fun quote(s: String) = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s

    private companion object {
        val FOLDERS = listOf("sharp", "blurry", "night", "not-race")
        val JPEG = setOf("jpg", "jpeg")
    }
}

/** Linear-interpolated percentile (p from 0 to 100) of [values] in any order; null when empty. */
internal fun percentile(values: List<Double>, p: Double): Double? {
    if (values.isEmpty()) return null
    val sorted = values.sorted()
    val rank = p / 100.0 * (sorted.size - 1)
    val lo = rank.toInt()
    val hi = minOf(lo + 1, sorted.size - 1)
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo)
}
