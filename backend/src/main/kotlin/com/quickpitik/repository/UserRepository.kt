package com.quickpitik.repository

import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository
import java.util.UUID

@Repository
interface UserRepository : JpaRepository<User, UUID> {
    fun findByEmail(email: String): User?

    fun existsByEmail(email: String): Boolean

    fun existsByRole(role: Role): Boolean
}
