package com.quickpitik.security

import com.quickpitik.entity.Role
import org.springframework.security.core.GrantedAuthority
import org.springframework.security.core.authority.SimpleGrantedAuthority
import org.springframework.security.core.userdetails.UserDetails
import java.util.UUID

class AuthPrincipal(
    val userId: UUID,
    val email: String,
    val role: Role,
    val suspended: Boolean = false,
) : UserDetails {
    override fun getAuthorities(): Collection<GrantedAuthority> =
        listOf(SimpleGrantedAuthority("ROLE_${role.name}"))

    override fun getPassword(): String = ""

    override fun getUsername(): String = email

    override fun isAccountNonExpired(): Boolean = true

    override fun isAccountNonLocked(): Boolean = true

    override fun isCredentialsNonExpired(): Boolean = true

    // Consulted by Spring's AccountStatusUserDetailsChecker on the
    // CustomUserDetailsService path. JwtAuthenticationFilter never builds a
    // suspended principal in the first place, so this is belt-and-braces.
    override fun isEnabled(): Boolean = !suspended
}
