package com.quickpitik.repository

import com.quickpitik.entity.Role
import com.quickpitik.entity.User
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import java.util.UUID

@Repository
interface UserRepository : JpaRepository<User, UUID> {
    // Serializes payout destination changes/requests and MFA proof consumption,
    // including the first account/enrollment when there is no child row to lock.
    @Lock(LockModeType.PESSIMISTIC_WRITE)
    @Query("SELECT u FROM User u WHERE u.id = :id")
    fun lockById(@Param("id") id: UUID): User?

    fun findByEmail(email: String): User?

    fun findByGoogleSub(googleSub: String): User?

    fun existsByEmail(email: String): Boolean

    fun existsByRole(role: Role): Boolean

    fun countByRole(role: Role): Long

    // Admin user list with optional role / search filters. Status filtering
    // (pending / approved / incomplete / suspended) is applied service-side
    // because it spans users + photographer_settings — keeping the JPQL flat
    // here avoids a multi-join expression that's hard to reason about.
    @Query(
        """
        SELECT u FROM User u
        WHERE (:role IS NULL OR u.role = :role)
          AND (:search = '' OR
               LOWER(u.name)  LIKE CONCAT('%', :search, '%') OR
               LOWER(u.email) LIKE CONCAT('%', :search, '%'))
        ORDER BY u.createdAt DESC, u.id ASC
        """,
        countQuery = """
        SELECT COUNT(u) FROM User u
        WHERE (:role IS NULL OR u.role = :role)
          AND (:search = '' OR
               LOWER(u.name)  LIKE CONCAT('%', :search, '%') OR
               LOWER(u.email) LIKE CONCAT('%', :search, '%'))
        """,
    )
    fun searchForAdmin(
        @Param("role") role: Role?,
        @Param("search") search: String,
        pageable: Pageable,
    ): Page<User>
}
