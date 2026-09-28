package com.quickpitik.service.image

import com.adobe.internal.xmp.XMPConst
import com.adobe.internal.xmp.XMPMeta
import com.drew.imaging.ImageMetadataReader
import com.drew.lang.Rational
import com.drew.metadata.Directory
import com.drew.metadata.exif.ExifIFD0Directory
import com.drew.metadata.exif.ExifSubIFDDirectory
import com.drew.metadata.iptc.IptcDirectory
import com.drew.metadata.xmp.XmpDirectory
import com.quickpitik.dto.photos.PhotoMetadataFields
import java.io.InputStream
import java.text.Normalizer
import java.time.LocalDateTime
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.time.format.ResolverStyle

/** Reads only display-safe metadata from an already bounded original image stream. */
class PhotoMetadataExtractor {
    fun extract(input: InputStream): PhotoMetadataFields {
        val metadata = ImageMetadataReader.readMetadata(input)
        val ifd0 = metadata.getFirstDirectoryOfType(ExifIFD0Directory::class.java)
        val exif = metadata.getFirstDirectoryOfType(ExifSubIFDDirectory::class.java)
        val iptc = metadata.getFirstDirectoryOfType(IptcDirectory::class.java)
        val xmp = metadata.getFirstDirectoryOfType(XmpDirectory::class.java)?.xmpMeta

        val make = clean(ifd0.string(ExifIFD0Directory.TAG_MAKE), 256) ?: clean(xmp.value(XMPConst.NS_TIFF, "Make"), 256)
        val rawModel = clean(ifd0.string(ExifIFD0Directory.TAG_MODEL), 256) ?: clean(xmp.value(XMPConst.NS_TIFF, "Model"), 256)
        val model = rawModel?.let { if (make != null && it.startsWith("$make ", ignoreCase = true)) it.drop(make.length + 1) else it }
        val lens = clean(exif.string(ExifSubIFDDirectory.TAG_LENS_MODEL), 256)
            ?: clean(xmp.value(XMPConst.NS_EXIFX, "LensModel"), 256)
            ?: clean(xmp.value(XMPConst.NS_EXIF_AUX, "Lens"), 256)
        val aperture = exif.rational(ExifSubIFDDirectory.TAG_FNUMBER)?.toDouble()?.takeIf { it.isFinite() && it > 0 }
            ?: xmpAperture(xmp.value(XMPConst.NS_EXIF, "FNumber"))
        val exposureTime = exposure(exif.rational(ExifSubIFDDirectory.TAG_EXPOSURE_TIME))
            ?: exposure(xmp.value(XMPConst.NS_EXIF, "ExposureTime"))
        val iso = exif.number(ExifSubIFDDirectory.TAG_ISO_EQUIVALENT)?.takeIf { it > 0 }
            ?: exif.number(ExifSubIFDDirectory.TAG_ISO_SPEED)?.takeIf { it > 0 }
            ?: exif.number(ExifSubIFDDirectory.TAG_STANDARD_OUTPUT_SENSITIVITY)?.takeIf { it > 0 }
            ?: xmp.value(XMPConst.NS_EXIF, "ISOSpeedRatings")?.toLongOrNull()?.takeIf { it > 0 }
            ?: xmp.arrayValue(XMPConst.NS_EXIF, "ISOSpeedRatings")?.toLongOrNull()?.takeIf { it > 0 }
        val author = clean(xmp.arrayValue(XMPConst.NS_DC, "creator"), 500)
            ?: clean(iptc.string(IptcDirectory.TAG_BY_LINE), 500)
            ?: clean(ifd0.string(ExifIFD0Directory.TAG_ARTIST), 500)
        val copyright = clean(xmp.localizedValue(XMPConst.NS_DC, "rights"), 1000)
            ?: clean(iptc.string(IptcDirectory.TAG_COPYRIGHT_NOTICE), 1000)
            ?: clean(ifd0.string(ExifIFD0Directory.TAG_COPYRIGHT), 1000)
        val taken = exifDate(exif) ?: xmpDate(xmp) ?: iptcDate(iptc)
        val flash = exif.number(ExifSubIFDDirectory.TAG_FLASH)?.takeIf { it in 0L..0x7fL }
            ?.let { if (it and 1L == 1L) "Fired" else "Did not fire" }
            ?: xmp.value(XMPConst.NS_EXIF, "Flash/exif:Fired")?.let {
                when (it.lowercase()) { "true", "1" -> "Fired"; "false", "0" -> "Did not fire"; else -> null }
            }
        val exposureMode = (exif.number(ExifSubIFDDirectory.TAG_EXPOSURE_MODE)?.takeIf { it in 0L..2L }
            ?: xmp.value(XMPConst.NS_EXIF, "ExposureMode")?.toLongOrNull())?.let {
            when (it) { 0L -> "Auto"; 1L -> "Manual"; 2L -> "Auto bracket"; else -> null }
        }
        val whiteBalance = (exif.number(ExifSubIFDDirectory.TAG_WHITE_BALANCE_MODE)?.takeIf { it in 0L..1L }
            ?: xmp.value(XMPConst.NS_EXIF, "WhiteBalance")?.toLongOrNull())?.let {
            when (it) { 0L -> "Auto"; 1L -> "Manual"; else -> null }
        }

        // Reserve ample space for JSON keys and escaping; values remain below 8 KiB serialized.
        var remaining = 7_000
        fun fit(value: String?): String? {
            if (value == null) return null
            val result = StringBuilder()
            for (point in value.codePoints().toArray()) {
                val unit = String(Character.toChars(point))
                val cost = when {
                    point > 0xffff -> 12 // Jackson writes a surrogate pair as two six-byte escapes.
                    point == '"'.code || point == '\\'.code -> 2
                    else -> unit.toByteArray(Charsets.UTF_8).size
                }
                if (cost > remaining) break
                result.append(unit)
                remaining -= cost
            }
            return result.toString().ifBlank { null }
        }
        return PhotoMetadataFields(
            cameraMake = fit(make), cameraModel = fit(model), lens = fit(lens), aperture = aperture,
            exposureTime = exposureTime, iso = iso, author = fit(author), copyright = fit(copyright),
            takenAtLocal = taken?.first, takenAtOffset = taken?.second,
            flash = flash, exposureMode = exposureMode, whiteBalance = whiteBalance,
        )
    }

    private fun clean(value: String?, limit: Int): String? = value?.let {
        Normalizer.normalize(it, Normalizer.Form.NFC)
            .filterNot { char -> Character.isISOControl(char) }
            .trim().codePoints().limit(limit.toLong()).toArray().let { points -> String(points, 0, points.size) }
            .ifBlank { null }
    }

    private fun Directory?.string(tag: Int): String? = runCatching { this?.getString(tag) }.getOrNull()?.takeIf { it.isNotBlank() }
    private fun Directory?.rational(tag: Int): Rational? = runCatching { this?.getRational(tag) }.getOrNull()
        ?.takeIf { it.denominator > 0L && it.numerator > 0L }
    private fun Directory?.number(tag: Int): Long? = runCatching { this?.getLong(tag) }.getOrNull()

    private fun xmpAperture(value: String?): Double? {
        val parts = value?.split('/') ?: return null
        val number = when (parts.size) {
            1 -> parts[0].trim().toDoubleOrNull()
            2 -> {
                val numerator = parts[0].trim().toLongOrNull() ?: return null
                val denominator = parts[1].trim().toLongOrNull() ?: return null
                if (numerator <= 0 || denominator <= 0) return null
                numerator.toDouble() / denominator
            }
            else -> null
        }
        return number?.takeIf { it.isFinite() && it > 0 }
    }

    private fun XMPMeta?.value(ns: String, name: String): String? =
        runCatching { this?.getPropertyString(ns, name) }.getOrNull()?.takeIf { it.isNotBlank() }

    private fun XMPMeta?.arrayValue(ns: String, name: String): String? = runCatching {
        if (this == null) return@runCatching null
        (1..countArrayItems(ns, name)).firstNotNullOfOrNull { getArrayItem(ns, name, it)?.value?.takeIf(String::isNotBlank) }
    }.getOrNull()

    private fun XMPMeta?.localizedValue(ns: String, name: String): String? = runCatching {
        if (this == null) return@runCatching null
        val default = getLocalizedText(ns, name, null, "x-default")?.value?.takeIf(String::isNotBlank)
        default ?: arrayValue(ns, name)
    }.getOrNull()

    private fun exposure(value: Rational?): String? = value?.let {
        val r = it.simplifiedInstance
        if (r.denominator == 1L) "${r.numerator} s" else "${r.numerator}/${r.denominator} s"
    }

    private fun exposure(value: String?): String? {
        if (value == null) return null
        val parts = value.trim().split('/')
        if (parts.size == 2) {
            val n = parts[0].trim().toLongOrNull() ?: return null
            val d = parts[1].trim().toLongOrNull() ?: return null
            return exposure(Rational(n, d).takeIf { d > 0L && n > 0L })
        }
        val seconds = value.toDoubleOrNull()?.takeIf { it.isFinite() && it > 0 } ?: return null
        return "${seconds.toBigDecimal().stripTrailingZeros().toPlainString()} s"
    }

    private fun exifDate(exif: ExifSubIFDDirectory?): Pair<LocalDateTime, String?>? = runCatching {
        val raw = exif.string(ExifSubIFDDirectory.TAG_DATETIME_ORIGINAL) ?: return@runCatching null
        val base = LocalDateTime.parse(raw, DateTimeFormatter.ofPattern("uuuu:MM:dd HH:mm:ss").withResolverStyle(ResolverStyle.STRICT))
        val fraction = exif.string(ExifSubIFDDirectory.TAG_SUBSECOND_TIME_ORIGINAL)
            ?.takeIf { it.matches(Regex("[0-9]{1,9}")) }
        val local = fraction?.let { base.withNano(it.padEnd(9, '0').toInt()) } ?: base
        val offset = exif.string(ExifSubIFDDirectory.TAG_TIME_ZONE_ORIGINAL)
            ?.let { runCatching { ZoneOffset.of(it).id }.getOrNull() }
        local to offset
    }.getOrNull()

    private fun xmpDate(xmp: XMPMeta?): Pair<LocalDateTime, String?>? = runCatching {
        val raw = xmp.value(XMPConst.NS_EXIF, "DateTimeOriginal") ?: return@runCatching null
        runCatching { OffsetDateTime.parse(raw).let { it.toLocalDateTime() to it.offset.id } }.getOrNull()
            ?: LocalDateTime.parse(raw).let { it to null }
    }.getOrNull()

    private fun iptcDate(iptc: IptcDirectory?): Pair<LocalDateTime, String?>? = runCatching {
        val date = iptc.string(IptcDirectory.TAG_DATE_CREATED) ?: return@runCatching null
        val time = iptc.string(IptcDirectory.TAG_TIME_CREATED) ?: return@runCatching null
        val local = LocalDateTime.parse(date + time.take(6), DateTimeFormatter.ofPattern("uuuuMMddHHmmss").withResolverStyle(ResolverStyle.STRICT))
        val offset = time.drop(6).takeIf(String::isNotBlank)?.let { runCatching { ZoneOffset.of(it).id }.getOrNull() }
        local to offset
    }.getOrNull()
}

fun PhotoMetadataFields.hasRecordedTags(): Boolean =
    cameraMake != null || cameraModel != null || lens != null || aperture != null || exposureTime != null || iso != null ||
        author != null || copyright != null || takenAtLocal != null || flash != null || exposureMode != null || whiteBalance != null
