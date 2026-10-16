package com.quickpitik.mobile

import android.content.Intent
import android.net.Uri
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.runComposeUiTest
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.navigation
import androidx.navigation.compose.rememberNavController
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * The Activity's routing decisions, now pure: where a cold start lands, where
 * the role guard bounces a wrong-role visitor, which links the app accepts and
 * what each one means, and the receipt route with its id sanitising; plus the
 * email-link exit, the one side effect here, run against a real NavHost.
 * Robolectric for android.net.Uri and that NavHost.
 */
@RunWith(RobolectricTestRunner::class)
class MainActivityRoutingTest {
    @Test fun `cold start lands on login, studio, or the runner view the photographer left in`() {
        assertEquals("login", startDestinationFor(signedIn = false, photographer = true, runnerView = true))
        assertEquals("studio", startDestinationFor(signedIn = true, photographer = true, runnerView = false))
        assertEquals("runner", startDestinationFor(signedIn = true, photographer = true, runnerView = true))
        assertEquals("runner", startDestinationFor(signedIn = true, photographer = false, runnerView = false))
    }

    @Test fun `the guard checks the effective role on runner routes and the true role on studio routes`() {
        // a runner
        assertNull(guardRedirect(photographer = false, runnerView = false, route = "events"))
        assertEquals("runner", guardRedirect(photographer = false, runnerView = false, route = "studio/home"))
        // a photographer in studio
        assertEquals("studio", guardRedirect(photographer = true, runnerView = false, route = "gallery"))
        assertEquals("studio", guardRedirect(photographer = true, runnerView = false, route = RECEIPT_ROUTE_PATTERN))
        assertNull(guardRedirect(photographer = true, runnerView = false, route = "studio/capture"))
        // a photographer browsing as a runner may still deep-return to studio
        assertNull(guardRedirect(photographer = true, runnerView = true, route = "gallery"))
        assertNull(guardRedirect(photographer = true, runnerView = true, route = "studio/capture"))
        // shared surfaces are in neither set
        for (route in listOf("photographer/{handle}", "login", "verify-email?token={token}")) {
            assertNull(route, guardRedirect(photographer = true, runnerView = false, route = route))
            assertNull(route, guardRedirect(photographer = false, runnerView = false, route = route))
        }
    }

    @Test fun `the receipt pattern is guarded exactly as registered and never shows the bottom bar`() {
        assertTrue(RECEIPT_ROUTE_PATTERN in RUNNER_ROUTES)
        assertFalse(RECEIPT_ROUTE_PATTERN in RUNNER_NAV_ROUTES)
        assertTrue("orders?orderId={orderId}" in RUNNER_NAV_ROUTES)
    }

    @Test fun `the studio bar needs the studio graph entry, the runner bar never shows on the receipt`() {
        assertEquals(BottomBar.Studio, bottomBarFor("studio/home", hasStudioEntry = true))
        assertEquals(BottomBar.Studio, bottomBarFor("studio/settings", hasStudioEntry = true))
        // Mid-teardown the graph entry is gone: no bar rather than a crash.
        assertEquals(BottomBar.None, bottomBarFor("studio/home", hasStudioEntry = false))
        assertEquals(BottomBar.Runner, bottomBarFor("events", hasStudioEntry = false))
        assertEquals(BottomBar.Runner, bottomBarFor("orders?orderId={orderId}", hasStudioEntry = true))
        // The modal receipt, login and a public profile carry no persistent nav.
        for (bare in listOf(RECEIPT_ROUTE_PATTERN, "login", "photographer/{handle}", "verify", "orders", null)) {
            assertEquals(bare.toString(), BottomBar.None, bottomBarFor(bare, hasStudioEntry = true))
        }
    }

    @Test fun `tab order ignores the query and slides only between peers of the same shell`() {
        assertEquals(2, tabIndexForRoute("orders?orderId={orderId}"))
        assertEquals(2, tabIndexForRoute("studio/events"))
        assertEquals(-1, tabIndexForRoute("gallery"))
        assertEquals(-1, tabIndexForRoute(null))
        assertTrue(arePeerTabs("events", "settings"))
        assertTrue(arePeerTabs("studio/home", "studio/settings"))
        assertFalse(arePeerTabs("events", "studio/home"))
        assertFalse(arePeerTabs(null, "events"))
    }

    @Test fun `the receipt route encodes both parts and refuses an id that is not route-safe`() {
        assertEquals("orders/return/abc-123?token=t1", receiptRoute("abc-123", "t1"))
        assertEquals("orders/return/abc-123", receiptRoute("abc-123", ""))
        assertEquals("orders/return/abc-123", receiptRoute("abc-123", null))
        assertEquals("orders/return/abc-123?token=a%20b", receiptRoute("abc-123", "a b"))
        for (bad in listOf(null, "", "  ", "../x", "a b", "x".repeat(65))) {
            assertNull(bad, receiptRoute(bad, "t"))
        }
    }

    @Test fun `event share defaults to active and earnings opens purchased`() {
        assertEquals("studio/share/event-a", shareRoute("event-a"))
        assertEquals("studio/share/event-a?tab=purchased", shareRoute("event-a", "purchased"))
    }

    @Test fun `only quickpitik links and the two email paths are accepted, and each maps to one target`() {
        assertNull(pickQuickpitikUri(null))
        assertNull(pickQuickpitikUri(Intent()))
        assertNull(pickQuickpitikUri(Intent(Intent.ACTION_VIEW, Uri.parse("https://quickpitik.com/events/cebu"))))
        val cart = Uri.parse("quickpitik://cart")
        assertSame(cart, pickQuickpitikUri(Intent(Intent.ACTION_VIEW, cart)))
        val verify = Uri.parse("https://quickpitik.com/verify-email/?token=v1")
        assertSame(verify, pickQuickpitikUri(Intent(Intent.ACTION_VIEW, verify)))

        assertEquals(DeepLinkTarget.VerifyEmail("v1"), deepLinkTarget(verify))
        assertEquals(DeepLinkTarget.ConfirmEmail("c1"), deepLinkTarget(Uri.parse("quickpitik://confirm-email-change?token=c1")))
        assertEquals(DeepLinkTarget.ConfirmEmail(""), deepLinkTarget(Uri.parse("https://quickpitik.com/confirm-email-change")))
        assertEquals(DeepLinkTarget.OrderReturn("o1", "t1"), deepLinkTarget(Uri.parse("quickpitik://orders/return?orderId=o1&token=t1")))
        assertEquals(DeepLinkTarget.OrderReturn(null, null), deepLinkTarget(Uri.parse("quickpitik://orders/return")))
        assertEquals(DeepLinkTarget.Orders, deepLinkTarget(Uri.parse("quickpitik://orders")))
        assertEquals(DeepLinkTarget.Cart, deepLinkTarget(cart))
        assertNull(deepLinkTarget(Uri.parse("quickpitik://studio")))
    }

    @Test fun `an email link whose token is too long is ignored`() {
        // A real token is 43 characters; a ~50 KB one crashed the app on Home (NV-1).
        val max = "a".repeat(256)
        assertEquals(DeepLinkTarget.ConfirmEmail(max), deepLinkTarget(Uri.parse("quickpitik://confirm-email-change?token=$max")))
        assertEquals(DeepLinkTarget.VerifyEmail(max), deepLinkTarget(Uri.parse("quickpitik://verify-email?token=$max")))
        val huge = "a".repeat(257)
        assertNull(deepLinkTarget(Uri.parse("quickpitik://confirm-email-change?token=$huge")))
        assertNull(deepLinkTarget(Uri.parse("https://quickpitik.com/verify-email?token=$huge")))
    }

    @Test fun `an order-return link whose token is too long is ignored`() {
        // A real return token is about 105 characters; any web page can send a huge one.
        val max = "a".repeat(256)
        assertEquals(DeepLinkTarget.OrderReturn("o1", max), deepLinkTarget(Uri.parse("quickpitik://orders/return?orderId=o1&token=$max")))
        val huge = "a".repeat(257)
        assertNull(deepLinkTarget(Uri.parse("quickpitik://orders/return?orderId=o1&token=$huge")))
    }

    @OptIn(ExperimentalTestApi::class)
    @Test fun `an email link's exit keeps a live studio when signed in, clears to login when signed out, and ignores a second tap`() {
        runComposeUiTest {
            lateinit var nav: NavHostController
            setContent {
                nav = rememberNavController()
                NavHost(nav, startDestination = "studio") {
                    composable("login") {}
                    navigation(startDestination = "studio/home", route = "studio") { composable("studio/home") {} }
                    composable("verify-email?token={token}") {}
                }
            }
            fun openLink(): NavBackStackEntry = runOnIdle {
                nav.navigate("verify-email?token=t") { launchSingleTop = true }
                nav.currentBackStackEntry!!
            }
            // The studio graph entry owns the tether's ViewModel.
            fun studioAlive() = runCatching { nav.getBackStackEntry("studio") }.isSuccess

            // Signed in: back to the studio. The second call is a double tap
            // landing while the screen animates out.
            val link = openLink()
            runOnIdle {
                nav.leaveEmailLink(link, signedIn = true)
                nav.leaveEmailLink(link, signedIn = true)
            }
            runOnIdle {
                assertEquals("studio/home", nav.currentDestination?.route)
                assertTrue(studioAlive())
            }

            // Signed out: nothing underneath is kept.
            val signedOut = openLink()
            runOnIdle { nav.leaveEmailLink(signedOut, signedIn = false) }
            runOnIdle {
                assertEquals("login", nav.currentDestination?.route)
                assertFalse(studioAlive())
            }
        }
    }
}
