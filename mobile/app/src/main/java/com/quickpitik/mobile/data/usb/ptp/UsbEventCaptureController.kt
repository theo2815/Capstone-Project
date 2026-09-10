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
    private val awaitDevice: suspend (Long) -> Unit = { delay(it) },
    private val openCamera: suspend (UsbDevice, Boolean) -> GenericPtpCamera = { device, warm ->
        CameraDrivers.open(manager, device, live = true, warmUp = warm)
    },
    private val nowMs: () -> Long = { System.nanoTime() / 1_000_000 },
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
        var lostAt: Long? = nowMs()
        var reconnectDelay = 2000L
        var warmedAttachment: String? = null
        val disabledEvents = HashSet<String>()
        while (currentCoroutineContext().isActive) {
            if (!isOwnerActive()) throw PtpException("Sign in to the account that started this shoot.")
            val device = deviceProvider()
            if (device == null) {
                onReconnecting(true)
                val since = lostAt ?: nowMs().also { lostAt = it }
                val remaining = RECONNECT_MS - (nowMs() - since)
                if (remaining <= 0) throw PtpException("Camera is still disconnected or needs USB permission. Reconnect and start again.")
                awaitDevice(minOf(remaining, 30_000))
                continue
            }
            try {
                val attachment = "${device.deviceName}:${device.deviceId}"
                openCamera(device, attachment != warmedAttachment).use { camera ->
                    warmedAttachment = attachment
                    val sessionStarted = nowMs()
                    fun healthy() {
                        // Flapping sessions must not restart the five-minute recovery budget.
                        if (nowMs() - sessionStarted >= 10_000) { lostAt = null; reconnectDelay = 2000 }
                    }
                    if (boundCamera != null && camera.cameraKey != boundCamera)
                        throw DifferentCamera()
                    boundCamera = camera.cameraKey
                    onLog("${camera.info.manufacturer} ${camera.info.model} (${camera.info.deviceVersion})")
                    onLog(camera.capabilities.physicalShutter)
                    onLog(CameraDrivers.usbGuidance(device.vendorId))
                    camera.capabilities.assumptions.forEach { onLog("Assumed, not verified on this body: $it") }
                    camera.setupWarning?.let(onLog)
                    if (camera.capabilities.transientObjects) {
                        watchTransient(camera, owner, eventId, isOwnerActive, onLog,
                            onReady = { if (!started) { started = true; onStarted() }; onReconnecting(false); healthy() },
                            onPaused, onCapture)
                        return@use
                    }
                    val scope = fingerprint(owner, eventId, camera.cameraKey)
                    val cache = LinkedHashMap<Long, CameraTransfer>()
                    val eventHandles = LinkedHashSet<Long>()
                    var checkIndex = 0
                    var lastKeepAlive = nowMs()
                    var lastMetadataCheck = nowMs()
                    var firstScan = true
                    var eventPolling = camera.capabilities.eventPolling && camera.cameraKey !in disabledEvents
                    var eventHits = 0
                    var eventMisses = 0
                    var formatMessage: String? = null
                    while (currentCoroutineContext().isActive) {
                        if (!isOwnerActive()) throw PtpException("Account changed. Start a new shoot after signing in.")
                        val handles = (camera.handles() + eventHandles).distinct().toMutableList()
                        val refresh = linkedSetOf<Long>()
                        var announced: List<Long> = emptyList()
                        if (eventPolling) {
                            try {
                                // Events prioritize metadata refresh, including reused handles; scans repair lost events.
                                announced = camera.events()
                                refresh.addAll(announced)
                            } catch (e: PtpResponseException) {
                                if (e.code != Ptp.RC_OPERATION_NOT_SUPPORTED) throw e
                                eventPolling = false
                                onLog("Camera event polling unavailable; continuing card reconciliation.")
                            } catch (e: CancellationException) { throw e }
                            catch (e: PtpException) {
                                disabledEvents.add(camera.cameraKey)
                                if (!camera.session.isUsable) throw e
                                eventPolling = false
                                onLog("Camera event data could not be read; continuing card reconciliation.")
                            }
                        }
                        refresh.addAll(handles.filter { it !in cache })
                        // Revalidate a bounded slice too: handles can be reused without a USB detach.
                        if (handles.isNotEmpty() && nowMs() - lastMetadataCheck >= 30_000) {
                            repeat(minOf(8, handles.size)) {
                                refresh.add(handles[checkIndex % handles.size]); checkIndex = (checkIndex + 1) % handles.size
                            }
                            lastMetadataCheck = nowMs()
                        }
                        val changed = ArrayList<CameraTransfer>()
                        val freshHandles = handles.filter { it !in cache }.toSet()
                        var newRaw = false
                        var newJpeg = false
                        for (handle in refresh) {
                            currentCoroutineContext().ensureActive()
                            val info = try { camera.objectInfo(handle) } catch (e: PtpResponseException) {
                                if (e.code != Ptp.RC_INVALID_OBJECT_HANDLE) throw e
                                handles.remove(handle); cache.remove(handle); eventHandles.remove(handle)
                                continue // A photographer may delete a photo between listing and metadata.
                            }
                            if (handle !in handles) {
                                // Some bodies lag in enumeration. Accept a card event once metadata identifies storage;
                                // vendor RAM-only events need their own transfer/acknowledgement protocol.
                                if (info.storageId == 0L || info.storageId == 0xffffffffL) continue
                                handles.add(handle)
                                eventHandles.add(handle)
                            }
                            if (handle in freshHandles && !info.isAssociation) {
                                if (info.isJpeg) {
                                    newJpeg = true
                                    if (eventPolling && !firstScan) { if (handle in announced) eventHits++ else eventMisses++ }
                                } else newRaw = true
                            }
                            val updated = store.transfer(owner, eventId, camera.cameraKey, handle, info)
                            if (cache[handle]?.key != updated.key) changed.add(updated)
                            cache[handle] = updated
                        }
                        cache.keys.retainAll(handles.toSet())
                        if (eventPolling && eventHits == 0 && eventMisses >= 10) {
                            // ponytail: fixed 10-shot probation; make it per-model if a body proves to report events late.
                            eventPolling = false
                            disabledEvents.add(camera.cameraKey)
                            onLog("Camera events reported none of the last 10 new photos; relying on card scans for this camera.")
                        }
                        val firstBaseline = firstScan && store.dao.baselineExists(scope) == 0
                        val rows = if (firstScan || changed.isNotEmpty()) store.reconcile(scope, changed, pendingOnly = true)
                            else store.dao.pendingForScope(scope)
                        firstScan = false
                        if (firstBaseline) onLog("Existing photos remain available in card import. New captures are monitored after this baseline.")
                        if (newJpeg) formatMessage = null
                        else if (newRaw && !firstBaseline) formatMessage = "RAW files stay on the camera. Select JPEG or RAW+JPEG for automatic uploads."
                        if (!started) { started = true; onStarted() }
                        onReconnecting(false)
                        var pauseMessage = formatMessage
                        val currentKeys = cache.values.mapTo(HashSet()) { it.key }
                        val waiting = rows.filter {
                            !it.sourceKey.startsWith("transient:") && it.status in setOf("PENDING", "FAILED", "TRANSFERRING")
                        }
                        for (record in waiting) {
                            currentCoroutineContext().ensureActive()
                            if (!isOwnerActive()) throw PtpException("Account changed; camera ingestion paused.")
                            if (record.key !in currentKeys) {
                                if (record.error != "Photo is no longer available at its camera location. Use card import to review.")
                                    store.fail(record, "Photo is no longer available at its camera location. Use card import to review.")
                                continue
                            }
                            if (record.retryAfter > System.currentTimeMillis()) continue
                            if (record.attempts >= CameraTransferStore.MAX_TRANSFER_ATTEMPTS) continue
                            try {
                                // Never replay a delayed retry against a reused handle with different metadata.
                                val source = store.transfer(owner, eventId, camera.cameraKey, record.handle, camera.objectInfo(record.handle))
                                if (source.key != record.key) {
                                    store.fail(record, "Photo changed at its camera location. Use card import to review.")
                                    cache.remove(record.handle)
                                    continue
                                }
                                store.save(record) { sink -> camera.readObject(record.handle, sink, record.expectedBytes) }
                                healthy()
                                onCapture(record.filename)
                                onLog("${record.filename} saved on phone and queued.")
                            } catch (e: CancellationException) { throw e }
                            catch (e: LowCameraStorage) {
                                pauseMessage = e.message; onLog(e.message.orEmpty()); break
                            } catch (e: Exception) {
                                if (store.dao.get(record.key)?.attempts == record.attempts)
                                    store.fail(record, e.message ?: "Camera transfer failed.")
                                onLog("${record.filename}: ${e.message}; retained for retry.")
                                if (!camera.session.isUsable) throw e
                            }
                            if (nowMs() - lastKeepAlive >= 30_000) { camera.keepAlive(); lastKeepAlive = nowMs() }
                        }
                        if (nowMs() - lastKeepAlive >= 30_000) { camera.keepAlive(); lastKeepAlive = nowMs() }
                        if (waiting.any { it.attempts >= CameraTransferStore.MAX_TRANSFER_ATTEMPTS })
                            pauseMessage = "Some photos need attention. Check the camera, then tap Retry; originals remain on the card."
                        onPaused(pauseMessage)
                        healthy()
                        delay(3000)
                    }
                }
            } catch (e: CancellationException) { throw e }
            catch (e: DifferentCamera) { throw e }
            catch (e: CameraSetupException) { throw e }
            catch (e: Exception) {
                onReconnecting(true)
                onLog("${e.message} Reopening the camera; pending files are preserved.")
                val since = lostAt ?: nowMs().also { lostAt = it }
                val remaining = RECONNECT_MS - (nowMs() - since)
                if (remaining <= 0) throw PtpException("Camera recovery timed out. Check USB mode, cable and permission, then start again.", e)
                delay(minOf(reconnectDelay, remaining))
                reconnectDelay = minOf(reconnectDelay * 2, 30_000)
            }
        }
    }

    private suspend fun watchTransient(
        camera: GenericPtpCamera, owner: String, event: String, isOwnerActive: () -> Boolean,
        onLog: (String) -> Unit, onReady: () -> Unit, onPaused: (String?) -> Unit, onCapture: (String) -> Unit,
    ) {
        onLog("Reading camera + PC JPEG copies. Interrupted buffer transfers require card import.")
        var idleDelay = 500L
        while (currentCoroutineContext().isActive) {
            if (!isOwnerActive()) throw CameraSetupException("Account changed. Start a new shoot after signing in.")
            val next = camera.nextCapture()
            onReady()
            if (next == null) { onPaused(null); delay(idleDelay); idleDelay = minOf(3000, idleDelay * 2); continue }
            idleDelay = 500
            val (handle, info) = next
            val expectedBytes = info.expectedSize
            if (expectedBytes == null || expectedBytes > CameraTransferStore.MAX_PUBLISH_BYTES)
                throw CameraSetupException("Select original JPEG transfers of at most 25 MiB on the camera, then start again.")
            try {
                store.saveTransient(owner, event, camera.cameraKey, handle, info) {
                    camera.readObject(handle, it, expectedBytes)
                }
                onCapture(info.filename)
                onLog("${info.filename} saved on phone and queued.")
                onPaused(null)
            } catch (e: CancellationException) { throw e }
            catch (e: LowCameraStorage) {
                onPaused(e.message)
                if (e.transferStarted) throw CameraSetupException(
                    "Phone storage filled during a camera buffer transfer. Free space, import the camera card backup, then restart auto-upload.")
                delay(2000)
            }
            catch (e: Exception) {
                // GetObject may have advanced the camera's RAM queue even if local persistence failed.
                throw CameraSetupException("Camera buffer transfer interrupted. Import the backup from the camera card, then restart auto-upload. ${e.message.orEmpty()}")
            }
        }
    }

    private class DifferentCamera : PtpException("Camera identity changed. Start again to select this camera; previous pending files are preserved.")
    private companion object { const val RECONNECT_MS = 5 * 60_000L }
}
