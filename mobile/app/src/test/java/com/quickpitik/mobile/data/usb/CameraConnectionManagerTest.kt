package com.quickpitik.mobile.data.usb

import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.usb.UsbConfiguration
import android.hardware.usb.UsbConstants
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.hardware.usb.UsbManager
import android.os.Looper
import android.os.Parcelable
import androidx.test.core.app.ApplicationProvider
import com.quickpitik.mobile.data.usb.CameraConnectionState.Connected
import com.quickpitik.mobile.data.usb.CameraConnectionState.Disconnected
import com.quickpitik.mobile.data.usb.CameraConnectionState.PermissionRequired
import com.quickpitik.mobile.data.usb.CameraConnectionState.Unavailable
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config
import org.robolectric.annotation.Implementation
import org.robolectric.annotation.Implements
import org.robolectric.shadow.api.Shadow
import org.robolectric.shadows.ShadowUsbManager
import org.robolectric.util.ReflectionHelpers
import org.robolectric.util.ReflectionHelpers.ClassParameter

private const val CANON = 0x04a9

/**
 * The attach → permission → connected state machine the Capture tab and the
 * tether service wait on: a plugged-in camera asks for USB permission once,
 * Deny leaves a denial the user can retry and Allow connects, unplugging
 * disconnects, the answer to a prompt from before a replug is ignored (its
 * generation is stale), and a Canon in storage mode reads differently from any
 * other USB device. Broadcasts reach the real receiver; ShadowUsbManager holds
 * the device list and the grants, and [PermissionPrompt] records the ask that
 * ShadowUsbManager does not implement.
 */
@RunWith(RobolectricTestRunner::class)
@Config(shadows = [CameraConnectionManagerTest.PermissionPrompt::class])
class CameraConnectionManagerTest {
    /** Stands in for the system dialog: records each permission ask so a test can answer it. */
    @Implements(UsbManager::class)
    class PermissionPrompt : ShadowUsbManager() {
        val asks = mutableListOf<Pair<UsbDevice, PendingIntent>>()
        @Implementation fun requestPermission(device: UsbDevice, pi: PendingIntent) { asks += device to pi }
    }

    private val context: Context = ApplicationProvider.getApplicationContext()
    private val usb: PermissionPrompt = Shadow.extract(context.getSystemService(Context.USB_SERVICE))
    private val manager = CameraConnectionManager(context)
    private val r6 = device("/dev/bus/usb/001/002", CANON, 0x32F5, "Canon EOS R6", UsbConstants.USB_CLASS_STILL_IMAGE)
    private val stick = device("/dev/bus/usb/001/003", 0x0781, 0x5567, "Cruzer Blade", UsbConstants.USB_CLASS_MASS_STORAGE)

    @Before fun usbHost() = shadowOf(context.packageManager).setSystemFeature(PackageManager.FEATURE_USB_HOST, true)
    @After fun stop() = manager.stop()

    @Test fun `a plugged-in camera asks for permission once and waits for the answer`() {
        manager.start()
        assertEquals(Disconnected, manager.state.value)

        plug(r6)
        assertEquals(PermissionRequired("Canon EOS R6"), manager.state.value)
        assertSame(r6, usb.asks.single().first)
        plug(stick) // another attach while the dialog is up asks nothing new
        assertEquals(PermissionRequired("Canon EOS R6"), manager.state.value)
        assertEquals(1, usb.asks.size)
        assertNull(manager.connectedCameraDevice())
    }

    @Test fun `deny leaves a denial the user can retry, allow connects`() {
        usb.addOrUpdateUsbDevice(r6, false)
        manager.start()
        answer(usb.asks.single(), granted = false)
        assertEquals(PermissionRequired("Canon EOS R6", denied = true), manager.state.value)
        assertEquals(1, usb.asks.size) // a denial never re-prompts on its own

        manager.refresh() // the Capture tab's Rescan
        assertEquals(PermissionRequired("Canon EOS R6"), manager.state.value)
        answer(usb.asks[1], granted = true)
        assertEquals(Connected("Canon EOS R6", CANON, 0x32F5), manager.state.value)
        assertSame(r6, manager.connectedCameraDevice())
    }

    @Test fun `unplugging the connected camera disconnects it`() {
        usb.addOrUpdateUsbDevice(r6, true) // permission kept from an earlier "always open QuickPitik"
        manager.start()
        assertEquals(Connected("Canon EOS R6", CANON, 0x32F5), manager.state.value)
        assertSame(r6, manager.connectedCameraDevice())
        assertTrue(usb.asks.isEmpty())

        unplug(r6)
        assertEquals(Disconnected, manager.state.value)
        assertNull(manager.connectedCameraDevice())
    }

    @Test fun `the answer to a prompt from before a replug is ignored`() {
        usb.addOrUpdateUsbDevice(r6, false)
        manager.start()
        val stale = usb.asks.single()
        unplug(r6)
        plug(r6) // same bus address, so only the prompt's generation tells the two answers apart
        assertEquals(2, usb.asks.size)

        answer(stale, granted = false)
        assertEquals(PermissionRequired("Canon EOS R6"), manager.state.value) // not flipped to denied
        answer(usb.asks[1], granted = true)
        assertTrue(manager.state.value is Connected)
    }

    @Test fun `a Canon in storage mode reads differently from any other USB device`() {
        shadowOf(context.packageManager).setSystemFeature(PackageManager.FEATURE_USB_HOST, false)
        manager.start()
        assertEquals(Unavailable("This phone does not report USB host support."), manager.state.value)

        shadowOf(context.packageManager).setSystemFeature(PackageManager.FEATURE_USB_HOST, true)
        plug(stick)
        assertEquals(Disconnected, manager.state.value)
        plug(device("/dev/bus/usb/001/004", CANON, 0x32F5, "Canon EOS R6", UsbConstants.USB_CLASS_MASS_STORAGE))
        val state = manager.state.value
        assertTrue("$state", state is Unavailable && state.message.startsWith("The attached device has no usable PTP USB interface."))
        assertTrue(usb.asks.isEmpty())
    }

    private fun plug(device: UsbDevice) {
        usb.addOrUpdateUsbDevice(device, false)
        broadcast(Intent(UsbManager.ACTION_USB_DEVICE_ATTACHED).putExtra(UsbManager.EXTRA_DEVICE, device))
    }

    private fun unplug(device: UsbDevice) {
        usb.removeUsbDevice(device)
        broadcast(Intent(UsbManager.ACTION_USB_DEVICE_DETACHED).putExtra(UsbManager.EXTRA_DEVICE, device))
    }

    /** What the system does on Allow or Deny: fire the ask's PendingIntent with the device filled in. */
    private fun answer(ask: Pair<UsbDevice, PendingIntent>, granted: Boolean) {
        val (device, reply) = ask
        if (granted) usb.addOrUpdateUsbDevice(device, true)
        reply.send(context, 0, Intent().putExtra(UsbManager.EXTRA_DEVICE, device).putExtra(UsbManager.EXTRA_PERMISSION_GRANTED, granted))
        shadowOf(Looper.getMainLooper()).idle()
    }

    private fun broadcast(intent: Intent) {
        context.sendBroadcast(intent)
        shadowOf(Looper.getMainLooper()).idle()
    }

    // UsbDevice's constructor is hidden; these are the fields its getters read.
    private fun device(name: String, vendorId: Int, productId: Int, product: String, usbClass: Int): UsbDevice =
        Shadow.newInstanceOf(UsbDevice::class.java).apply {
            ReflectionHelpers.setField(this, "mName", name)
            ReflectionHelpers.setField(this, "mVendorId", vendorId)
            ReflectionHelpers.setField(this, "mProductId", productId)
            ReflectionHelpers.setField(this, "mProductName", product)
            ReflectionHelpers.setField(this, "mConfigurations", emptyArray<UsbConfiguration>())
            ReflectionHelpers.setField(this, "mInterfaces", arrayOf(bulkInterface(usbClass)))
        }

    private fun bulkInterface(usbClass: Int): UsbInterface {
        val int = Int::class.javaPrimitiveType!!
        val iface = ReflectionHelpers.callConstructor(UsbInterface::class.java,
            ClassParameter.from(int, 0), ClassParameter.from(int, 0), ClassParameter.from(String::class.java, null),
            ClassParameter.from(int, usbClass), ClassParameter.from(int, 1), ClassParameter.from(int, 1))
        val endpoints = listOf(0x81, 0x02).map { address -> // bulk IN, bulk OUT
            ReflectionHelpers.callConstructor(UsbEndpoint::class.java,
                ClassParameter.from(int, address), ClassParameter.from(int, UsbConstants.USB_ENDPOINT_XFER_BULK),
                ClassParameter.from(int, 512), ClassParameter.from(int, 0))
        }
        ReflectionHelpers.callInstanceMethod<Unit>(iface, "setEndpoints",
            ClassParameter.from(Array<Parcelable>::class.java, endpoints.toTypedArray<Parcelable>()))
        return iface
    }
}
