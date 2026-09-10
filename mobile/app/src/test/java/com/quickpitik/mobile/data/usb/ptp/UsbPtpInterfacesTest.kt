package com.quickpitik.mobile.data.usb.ptp

import android.hardware.usb.UsbEndpoint
import android.hardware.usb.UsbInterface
import android.os.Parcelable
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.util.ReflectionHelpers
import org.robolectric.util.ReflectionHelpers.ClassParameter

@RunWith(RobolectricTestRunner::class)
class UsbPtpInterfacesTest {
    @Test fun `composite discovery skips incomplete interface and prefers a usable default alternate`() {
        val incomplete = iface(0, 6, listOf(endpoint(0x81)))
        val alternate = iface(1, 6, listOf(endpoint(0x81), endpoint(0x02)))
        val default = iface(0, 6, listOf(endpoint(0x83), endpoint(0x04)))
        assertSame(default, UsbPtpInterfaces.find(listOf(incomplete, alternate, default)))
        assertSame(alternate, UsbPtpInterfaces.find(listOf(incomplete, alternate)))
    }

    @Test fun `storage webcam and arbitrary vendor interfaces are not probed as PTP`() {
        for (kind in listOf(8, 14, 255))
            assertNull(UsbPtpInterfaces.find(listOf(iface(0, kind, listOf(endpoint(0x81), endpoint(0x02))))))
        assertNull(UsbPtpInterfaces.find(listOf(iface(0, 6, emptyList()))))
    }

    @Test fun `vendor-class bulk interface is accepted only for a trusted camera device`() {
        val vendor = iface(0, 255, listOf(endpoint(0x81), endpoint(0x02)))
        assertSame(vendor, UsbPtpInterfaces.findVendorSpecific(listOf(vendor), trusted = true))
        assertNull(UsbPtpInterfaces.findVendorSpecific(listOf(vendor), trusted = false))
        assertNull(UsbPtpInterfaces.findVendorSpecific(listOf(iface(0, 255, listOf(endpoint(0x81)))), trusted = true))
        assertNull(UsbPtpInterfaces.find(listOf(vendor)))
    }

    private fun endpoint(address: Int): UsbEndpoint = ReflectionHelpers.callConstructor(UsbEndpoint::class.java,
        ClassParameter.from(Int::class.javaPrimitiveType!!, address),
        ClassParameter.from(Int::class.javaPrimitiveType!!, 2),
        ClassParameter.from(Int::class.javaPrimitiveType!!, 512),
        ClassParameter.from(Int::class.javaPrimitiveType!!, 0))

    private fun iface(alternate: Int, kind: Int, endpoints: List<UsbEndpoint>): UsbInterface {
        val value = ReflectionHelpers.callConstructor(UsbInterface::class.java,
            ClassParameter.from(Int::class.javaPrimitiveType!!, 1),
            ClassParameter.from(Int::class.javaPrimitiveType!!, alternate),
            ClassParameter.from(String::class.java, "camera"),
            ClassParameter.from(Int::class.javaPrimitiveType!!, kind),
            ClassParameter.from(Int::class.javaPrimitiveType!!, 1),
            ClassParameter.from(Int::class.javaPrimitiveType!!, 1))
        ReflectionHelpers.callInstanceMethod<Unit>(value, "setEndpoints",
            ClassParameter.from(Array<Parcelable>::class.java, endpoints.toTypedArray<Parcelable>()))
        return value
    }
}
