package com.quickpitik.controller

import jakarta.servlet.http.HttpServletResponse
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

// PayMongo redirects the user's mobile browser here after a successful charge
// (or a cancel) for orders created with clientPlatform="android". The endpoint
// returns a tiny HTML page that immediately hands off to the `quickpitik://`
// custom-scheme deep link the Android app registers — Chrome Custom Tabs
// auto-closes when the OS routes the scheme into the app, so the user lands
// back in OrdersScreen (or the cart on cancel) without ever seeing the website.
// A fallback button covers desktops / phones without the app installed.
@RestController
@RequestMapping("/api/v1/orders")
class MobileReturnController {

    @GetMapping("/mobile-return", produces = [MediaType.TEXT_HTML_VALUE])
    fun mobileReturn(
        @RequestParam orderId: String,
        @RequestParam(required = false) token: String?,
        response: HttpServletResponse,
    ) {
        val deepLink = buildDeepLink(path = "orders/return", orderId = orderId, token = token)
        writeBridgeHtml(response, deepLink, heading = "Payment confirmed")
    }

    @GetMapping("/mobile-cancel", produces = [MediaType.TEXT_HTML_VALUE])
    fun mobileCancel(
        @RequestParam(required = false) orderId: String?,
        response: HttpServletResponse,
    ) {
        val deepLink = buildDeepLink(path = "cart", orderId = orderId, token = null)
        writeBridgeHtml(response, deepLink, heading = "Returning to your cart")
    }

    private fun buildDeepLink(path: String, orderId: String?, token: String?): String {
        val params = buildList {
            orderId?.takeIf { it.isNotBlank() }?.let { add("orderId=" + encode(it)) }
            token?.takeIf { it.isNotBlank() }?.let { add("token=" + encode(it)) }
        }
        val query = if (params.isEmpty()) "" else "?" + params.joinToString("&")
        return "quickpitik://$path$query"
    }

    private fun writeBridgeHtml(response: HttpServletResponse, deepLink: String, heading: String) {
        response.contentType = "text/html; charset=UTF-8"
        response.characterEncoding = "UTF-8"
        val escapedLink = htmlEscape(deepLink)
        val escapedHeading = htmlEscape(heading)
        val html = """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8" />
              <meta name="viewport" content="width=device-width,initial-scale=1" />
              <title>QuickPitik · Returning to app</title>
              <style>
                html, body { margin: 0; padding: 0; background: #F4EFE4; color: #15171A; font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; }
                .wrap { min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 32px; text-align: center; }
                h1 { font-size: 20px; font-weight: 700; margin: 0 0 8px; }
                p { font-size: 14px; color: #5B6168; margin: 0 0 24px; max-width: 320px; line-height: 1.4; }
                a.button { display: inline-block; padding: 14px 24px; background: #19A47B; color: #F4EFE4; text-decoration: none; border-radius: 9999px; font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
              </style>
            </head>
            <body>
              <div class="wrap">
                <h1>$escapedHeading</h1>
                <p>Returning you to the QuickPitik app. If it doesn't open automatically, tap the button below.</p>
                <a class="button" href="$escapedLink">Open the app</a>
              </div>
              <script>
                // Fire the custom-scheme intent right after render. Chrome Custom Tabs
                // (and most mobile browsers) hand the URL to the OS, which routes it
                // to the installed QuickPitik app via the intent-filter; the tab then
                // auto-closes. On desktop / no app installed, the fallback button stays.
                setTimeout(function () { window.location.href = "$escapedLink"; }, 100);
              </script>
            </body>
            </html>
        """.trimIndent()
        response.writer.write(html)
        response.writer.flush()
    }

    private fun encode(value: String): String =
        URLEncoder.encode(value, StandardCharsets.UTF_8)

    private fun htmlEscape(value: String): String = value
        .replace("&", "&amp;")
        .replace("\"", "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
}
