package com.quickpitik.security

import jakarta.servlet.FilterChain
import jakarta.servlet.http.HttpServletRequest
import jakarta.servlet.http.HttpServletResponse
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.http.HttpHeaders
import org.springframework.stereotype.Component
import org.springframework.web.filter.OncePerRequestFilter
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

// Forces a "download, don't display inline" disposition on LocalFs storage
// responses when the URL carries `?disposition=attachment[&filename=…]`.
//
// Why this exists: <a download> only works for SAME-ORIGIN clicks. Links
// in email open cross-origin and the browser ignores the attribute,
// defaulting to inline display for image/jpeg. Setting
//   Content-Disposition: attachment; filename="…"
// on the server response makes the browser trigger a download regardless.
// S3 has native support via `response-content-disposition` on presigned
// URLs — this filter is the equivalent for the dev LocalFs backend so the
// email behaviour matches dev and prod.
//
// Filename hygiene: only ASCII letters/digits/-_.space survive. RFC 6266
// `filename*` carries the UTF-8 form for browsers that respect it.
@Component
@ConditionalOnProperty(
    prefix = "app.storage",
    name = ["backend"],
    havingValue = "LOCAL",
    matchIfMissing = true,
)
class StorageDownloadDispositionFilter : OncePerRequestFilter() {
    override fun shouldNotFilter(request: HttpServletRequest): Boolean =
        !request.requestURI.startsWith("/storage/")

    override fun doFilterInternal(
        request: HttpServletRequest,
        response: HttpServletResponse,
        filterChain: FilterChain,
    ) {
        if (request.getParameter("disposition")
                .equals("attachment", ignoreCase = true)
        ) {
            val name = sanitize(request.getParameter("filename"))
                ?: defaultName(request.requestURI)
            val encoded = URLEncoder.encode(name, StandardCharsets.UTF_8).replace("+", "%20")
            response.setHeader(
                HttpHeaders.CONTENT_DISPOSITION,
                "attachment; filename=\"$name\"; filename*=UTF-8''$encoded",
            )
        }
        filterChain.doFilter(request, response)
    }

    private fun sanitize(raw: String?): String? {
        if (raw.isNullOrBlank()) return null
        val cleaned = raw.replace(SAFE_FILE_CHARS, "_").trim('_', ' ', '.')
        return cleaned.take(120).ifBlank { null }
    }

    private fun defaultName(uri: String): String {
        val last = uri.substringAfterLast('/').substringBefore('?')
        return last.ifBlank { "quickpitik-photo.jpg" }
    }

    private companion object {
        val SAFE_FILE_CHARS = Regex("[^a-zA-Z0-9._\\- ]")
    }
}
