package com.quickpitik.dto.auth

data class AuthResponse(
    val accessToken: String,
    val refreshToken: String,
    val user: UserDto,
    // Opt-in admin MFA challenge: when true, accessToken/refreshToken are empty
    // and the client must POST /auth/mfa/verify with `mfaToken` + a TOTP/recovery
    // code to obtain the real tokens. Additive — existing clients read accessToken.
    val mfaRequired: Boolean = false,
    val mfaToken: String? = null,
)
