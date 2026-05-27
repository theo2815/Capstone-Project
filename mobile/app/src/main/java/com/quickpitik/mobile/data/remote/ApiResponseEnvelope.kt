package com.quickpitik.mobile.data.remote

// Backend wraps every response in `{ success, data, errors? }` where `errors`
// is a structured array (see backend/common/ApiResponse.kt). The legacy single
// `error: String?` field never matched the wire — Gson always left it null,
// so every callsite silently fell through to its `?: "fallback"` string and
// the real backend error message was lost.
//
// Wire-shape field is now `errors: List<ApiError>?`. `error` is kept as a
// computed getter so the ~50 existing `response.error ?: "..."` callsites
// continue to work without edit, and now actually surface the backend's
// message when one is present.
data class ApiResponseEnvelope<T>(
    val success: Boolean,
    val data: T? = null,
    val errors: List<ApiError>? = null,
) {
    val error: String? get() = errors?.firstOrNull()?.message
}
