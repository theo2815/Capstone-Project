package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import com.quickpitik.mobile.data.local.CameraTransfer
import com.quickpitik.mobile.data.local.CameraTransferStore
import com.quickpitik.mobile.data.local.LowCameraStorage
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive

/** Physical-shutter ingestion. Card reconciliation remains the completeness detector. */
class UsbEventCaptureController(
    private val manager: UsbManager,
    private val deviceProvider: () -> UsbDevice?,
    private val store: CameraTransferStore,
) {
    suspend fun run(
        owner: String,
        eventId: String,
        isOwnerActive: () -> Boolean,
        onLog: (String) -> Unit,
        onStarted: () -> Unit,
        onReconnecting: (Boolean) -> Unit,
        onPaused: (String?) -> Unit,
        onCapture: (String) -> Unit,
    ) {
        store.recover(owner)
        var boundCamera: String? = null
        var started = false
        var lostAt = System.nanoTime()
        val disabledEvents = HashSet<String>()
        while (currentCoroutineContext().isActive) {
            if (!isOwnerActive()) throw PtpException("Sign in to the account that started this shoot.")
            val device = deviceProvider()
            if (device == null) {
                onReconnecting(true)
                if (elapsedMs(lostAt) > RECONNECT_MS) throw PtpException("Camera is still disconnected or needs USB permission. Reconnect and start again.")
                delay(2000)
                continue
            }
            try {
                CameraDrivers.open(manager, device, live = true).use { camera ->
                    if (boundCamera != null && camera.cameraKey != boundCamera)
                        throw DifferentCamera()
                    boundCamera = camera.cameraKey
                    onLog("${camera.info.manufacturer} ${camera.info.model} (${camera.info.deviceVersion})")
                    onLog(camera.capabilities.physicalShutter)
                    val scope = fingerprint(owner, eventId, camera.cameraKey)
                    val cache = LinkedHashMap<Long, CameraTransfer>()
                    var checkIndex = 0
                    var lastKeepAlive = System.nanoTime()
                    var lastEvents = System.nanoTime()
                    var eventPolling = camera.capabilities.eventPolling && camera.cameraKey !in disabledEvents
                    var formatMessage: String? = null
                    while (currentCoroutineContext().isActive) {
                        if (!isOwnerActive()) throw PtpException("Account changed. Start a new shoot after signing in.")
                        val handles = camera.handles()
                        val refresh = handles.filter { it !in cache }.toMutableSet()
                        // Revalidate a bounded slice too: handles can be reused without a USB detach.
                        if (handles.isNotEmpty()) repeat(minOf(8, handles.size)) {
                            refresh.add(handles[checkIndex % handles.size]); checkIndex = (checkIndex + 1) % handles.size
                        }
                        val freshHandles = handles.filter { it !in cache }.toSet()
                        var newRaw = false
                        var newJpeg = false
                        for (handle in refresh) {
                            currentCoroutineContext().ensureActive()
                            val info = camera.objectInfo(handle)
                            if (handle in freshHandles && !info.isAssociation) { if (info.isJpeg) newJpeg = true else newRaw = true }
                            cache[handle] = store.transfer(owner, eventId, camera.cameraKey, handle, info)
                        }
                        cache.keys.retainAll(handles.toSet())
                        val firstBaseline = store.dao.baselineExists(scope) == 0
                        val rows = store.reconcile(scope, cache.values.toList())
                        if (firstBaseline) onLog("Existing photos remain available in card import. New captures are monitored after this baseline.")
                        if (newJpeg) formatMessage = null
                        else if (newRaw && !firstBaseline) formatMessage = "RAW files stay on the camera. Select JPEG or RAW+JPEG for automatic uploads."
                        if (!started) { started = true; onStarted() }
                        onReconnecting(false)
                        var pauseMessage = formatMessage
                        val currentKeys = cache.values.mapTo(HashSet()) { it.key }
                        val waiting = rows.filter { it.status in setOf("PENDING", "FAILED", "TRANSFERRING") }
                        for (record in waiting) {
                            currentCoroutineContext().ensureActive()
                            if (!isOwnerActive()) throw PtpException("Account changed; camera ingestion paused.")
                            if (record.key !in currentKeys) {
                                if (record.error != "Photo is no longer available at its camera location. Use card import to review.")
                                    store.fail(record, "Photo is no longer available at its camera location. Use card import to review.")
                                continue
                            }
                            if (record.retryAfter > System.currentTimeMillis()) continue
                            try {
                                store.save(record) { sink -> camera.readObject(record.handle, sink, record.expectedBytes) }
                                onCapture(record.filename)
                                onLog("${record.filename} saved on phone and queued.")
                            } catch (e: CancellationException) { throw e }
                            catch (e: LowCameraStorage) {
                                pauseMessage = e.message; onLog(e.message.orEmpty()); break
                            } catch (e: Exception) {
                                onLog("${record.filename}: ${e.message}; retained for retry.")
                                if (!camera.session.isUsable) throw e
                            }
                            if (elapsedMs(lastKeepAlive) >= 30_000) { camera.keepAlive(); lastKeepAlive = System.nanoTime() }
                        }
                        if (eventPolling && elapsedMs(lastEvents) >= 5000) {
                            try {
                                // Hints trigger the next reconciliation. Never consume/acknowledge a file before durable storage.
                                if (camera.events().isNotEmpty()) onLog("Camera reported new files; reconciling card.")
                            } catch (e: PtpResponseException) {
                                if (e.code != 0x2005) throw e
                                eventPolling = false
                                onLog("Event polling unavailable; continuing card reconciliation.")
                            } catch (e: CancellationException) { throw e }
                            catch (e: PtpException) {
                                disabledEvents.add(camera.cameraKey)
                                throw e
                            }
                            lastEvents = System.nanoTime()
                        }
                        if (elapsedMs(lastKeepAlive) >= 30_000) { camera.keepAlive(); lastKeepAlive = System.nanoTime() }
                        if (waiting.any { it.status == "FAILED" }) onLog("Some camera files need retry. Pending records are preserved.")
                        onPaused(pauseMessage)
                        lostAt = System.nanoTime()
                        delay(3000)
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: DifferentCamera) { throw e }
            catch (e: Exception) {
                onReconnecting(true)
                onLog("${e.message} Reopening the camera; pending files are preserved.")
                if (elapsedMs(lostAt) >= RECONNECT_MS) throw PtpException("Camera recovery timed out. Check USB mode, cable and permission, then start again.", e)
                delay(2000)
            }
        }
    }

    private class DifferentCamera : PtpException("Camera identity changed. Start again to select this camera; previous pending files are preserved.")
    private fun elapsedMs(since: Long) = (System.nanoTime() - since) / 1_000_000
    private companion object { const val RECONNECT_MS = 5 * 60_000L }
}
