package com.quickpitik.mobile.ui.common

import java.time.OffsetDateTime
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

// Money and backend timestamps with the region and zone pinned: a phone set to
// another language or time zone still reads ₱1,234.50 and Manila's clock; the
// website uses the browser's zone, which is Manila for its users, so a receipt, the
// earnings list and the website agree on the same sale. Each screen keeps its
// own decimals and date pattern.

/** "₱1,234.50"; [decimals] keeps each screen's precision (whole-peso tiles pass 0). */
fun peso(amount: Double, decimals: Int = 2): String =
    String.format(Locale.US, "₱%,.${decimals}f", amount)

/** "1,234" regardless of the phone's region, like peso(). Int and Long: photo/sales counts are both. */
fun count(n: Int): String = String.format(Locale.US, "%,d", n)
fun count(n: Long): String = String.format(Locale.US, "%,d", n)

internal val MANILA: ZoneId = ZoneId.of("Asia/Manila")

// DateTimeFormatter is immutable and thread-safe; cache one per pattern so a
// LazyColumn row does not rebuild it on every recomposition (the old code held statics).
private val FORMATTERS = ConcurrentHashMap<String, DateTimeFormatter>()

/**
 * A backend timestamp (ISO-8601 with an offset, UTC in practice) on Manila's
 * clock, in [pattern]. Null when missing or unreadable, so each screen keeps
 * its own placeholder.
 */
fun formatManila(iso: String?, pattern: String): String? {
    if (iso.isNullOrBlank()) return null
    val format = FORMATTERS.computeIfAbsent(pattern) { DateTimeFormatter.ofPattern(it, Locale.US) }
    return try {
        OffsetDateTime.parse(iso).atZoneSameInstant(MANILA).format(format)
    } catch (_: DateTimeParseException) {
        null
    }
}

/** "AUG 14, 2026" in both inboxes; the year matters on messages kept for months. */
fun formatInboxDate(iso: String): String = formatManila(iso, "MMM d, yyyy")?.uppercase() ?: iso
