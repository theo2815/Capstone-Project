package com.quickpitik.service.ratelimit

import jakarta.servlet.http.HttpServletRequest

// Rate-limit identity for pre-auth endpoints. Production sits behind Cloudflare, which
// sets CF-Connecting-IP on every proxied request (a client cannot inject it through the
// proxy). remoteAddr is the fallback — and NOTE that server.forward-headers-strategy=framework
// makes remoteAddr the FIRST X-Forwarded-For hop, which the client controls; OPS-3 closes
// the direct-to-origin path that could forge either header.
// ponytail: no trusted-proxy CIDR list — Railway's hops are not published.
fun clientIp(request: HttpServletRequest): String =
    request.getHeader("CF-Connecting-IP")?.trim()?.takeIf { it.isNotEmpty() }
        ?: request.remoteAddr
        ?: "unknown"
