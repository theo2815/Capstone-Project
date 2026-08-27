package com.quickpitik.service.photographer

import com.quickpitik.service.storage.StorageService
import io.micrometer.core.instrument.MeterRegistry
import org.springframework.stereotype.Service
import java.time.Duration
import java.time.Instant
import java.util.concurrent.ConcurrentHashMap

// Per-key TTL memo of photographer watermark logos. Every upload composites the
// photographer's logo, so without this the same small object is fetched from
// object storage once per photo — up to 600 identical GETs/min at the upload
// bucket's ceiling. Replacing a watermark stores it under a NEW key
// (PhotographerSettingsService.uploadWatermark deletes the previous key), so a
// stale entry can never be served for a replaced logo; the TTL only bounds
// memory held by dead keys.
@Service
class WatermarkLogoCache(
    private val storageService: StorageService,
    private val meterRegistry: MeterRegistry,
) {
    private val cache = ConcurrentHashMap<String, Entry>()

    fun get(key: String): ByteArray {
        val hit = cache[key]?.takeIf { Duration.between(it.loadedAt, Instant.now()) < TTL }
        if (hit != null) {
            meterRegistry.counter("qp.watermark.cache", "result", "hit").increment()
            return hit.bytes
        }
        meterRegistry.counter("qp.watermark.cache", "result", "miss").increment()
        val bytes = storageService.getBytes(key)
        // ponytail: crude size bound — clear everything past the cap instead of
        // LRU; entries repopulate on the next upload. Swap for a real cache if
        // active photographers ever exceed a few hundred.
        if (cache.size >= MAX_ENTRIES) cache.clear()
        cache[key] = Entry(bytes, Instant.now())
        return bytes
    }

    private class Entry(val bytes: ByteArray, val loadedAt: Instant)

    private companion object {
        val TTL: Duration = Duration.ofMinutes(5)
        const val MAX_ENTRIES = 200
    }
}
