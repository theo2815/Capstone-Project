package com.quickpitik.service.image

import java.util.UUID

// Photographer credit as XMP metadata inside the preview JPEG. Survives a
// download / save-as / Drive copy (screenshots and most social platforms strip
// it — the pixel mark and the pHash registry cover those). Byte-level APP1
// insertion: ImageIO has no XMP writer and a dependency for ~30 lines is not
// worth it.
object JpegXmp {

    fun creditPacket(name: String, handle: String?, photoId: UUID, year: Int): String {
        val creator = esc(name)
        val rights = esc("© $year $name · QuickPitik")
        val handleAttr = handle?.let { """ quickpitik:photographerHandle="${esc(it)}"""" } ?: ""
        // Machine-readable rights: plus:DataMining is the IPTC/PLUS AI opt-out
        // that crawlers and compliant tools read; UsageTerms and the IPTC
        // "special instructions" field carry the notice the pixels also show.
        val terms = esc(
            "Watermarked preview \u00A9 $year $name \u00B7 QuickPitik. All rights reserved. $INSTRUCTIONS Licence: https://quickpitik.com/verify",
        )
        return "<?xpacket begin=\"\uFEFF\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n" +
            """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/" xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/" xmlns:plus="http://ns.useplus.org/ldf/xmp/1.0/" xmlns:quickpitik="https://quickpitik.com/ns/1.0/" photoshop:Credit="QuickPitik" photoshop:Instructions="$INSTRUCTIONS" xmpRights:Marked="True" xmpRights:WebStatement="https://quickpitik.com/verify" plus:DataMining="http://ns.useplus.org/ldf/vocab/DMI-PROHIBITED-AIGENAI" quickpitik:photoId="$photoId"$handleAttr>
<dc:creator><rdf:Seq><rdf:li>$creator</rdf:li></rdf:Seq></dc:creator>
<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">$rights</rdf:li></rdf:Alt></dc:rights>
<xmpRights:UsageTerms><rdf:Alt><rdf:li xml:lang="x-default">$terms</rdf:li></rdf:Alt></xmpRights:UsageTerms>
</rdf:Description></rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>"""
    }

    // Inserts the packet right after SOI — or after the JFIF APP0 when there is
    // one, since the JFIF spec wants APP0 first. Readers (browsers, exiftool,
    // metadata-extractor) accept either order; we stay spec-clean anyway.
    fun inject(jpeg: ByteArray, xmp: String): ByteArray {
        require(jpeg.size >= 4 && jpeg[0] == MARKER && jpeg[1] == 0xD8.toByte()) { "not a JPEG" }
        var cut = 2
        if (jpeg[2] == MARKER && jpeg[3] == 0xE0.toByte()) {
            val app0Len = ((jpeg[4].toInt() and 0xFF) shl 8) or (jpeg[5].toInt() and 0xFF)
            cut = 4 + app0Len
        }
        val payload = XMP_PREAMBLE.toByteArray(Charsets.ISO_8859_1) + xmp.toByteArray(Charsets.UTF_8)
        val len = payload.size + 2
        require(len <= 0xFFFF) { "XMP packet too large for one APP1 segment" }
        val segment = byteArrayOf(MARKER, 0xE1.toByte(), (len shr 8).toByte(), (len and 0xFF).toByte()) + payload
        return jpeg.copyOfRange(0, cut) + segment + jpeg.copyOfRange(cut, jpeg.size)
    }

    private fun esc(s: String): String =
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")

    private const val INSTRUCTIONS = "Do not remove, alter or obscure the watermark or attribution."
    private const val MARKER = 0xFF.toByte()
    private const val XMP_PREAMBLE = "http://ns.adobe.com/xap/1.0/\u0000"
}
