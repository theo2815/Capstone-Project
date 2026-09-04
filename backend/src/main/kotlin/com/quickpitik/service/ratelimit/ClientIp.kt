package com.quickpitik.service.ratelimit

import jakarta.servlet.http.HttpServletRequest

// Rate-limit identity for pre-auth endpoints. Deliberately request.remoteAddr,
// NOT X-Forwarded-For: that header is client-forgeable, so trusting its first
// hop let an attacker mint a fresh bucket (and a fresh identity) per request.
// Behind a TLS-terminating proxy, configure Tomcat's RemoteIpValve instead —
//   server.tomcat.remoteip.remote-ip-header: X-Forwarded-For
//   server.tomcat.remoteip.internal-proxies: <trusted proxy IP regex>
// — which rewrites remoteAddr (and isSecure, arming HSTS) from the trusted
// hop only. Replaces three copy-pasted first-hop-XFF helpers (2026-08-27).
fun clientIp(request: HttpServletRequest): String = request.remoteAddr ?: "unknown"
