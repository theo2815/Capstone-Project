package com.quickpitik.repository

import com.quickpitik.entity.Role
import com.quickpitik.entity.SellerDetailsStatus
import com.quickpitik.entity.User
import jakarta.persistence.LockModeType
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Lock
import org.springframework.data.jpa.repository.Modifying
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import org.springframework.stereotype.Repository
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDate
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

    // AA-I12a: which of these users are suspended (PhotographerSuspension).
    @Query("SELECT u.id FROM User u WHERE u.id IN :ids AND u.suspendedAt IS NOT NULL")
    fun findSuspendedIdsIn(@Param("ids") ids: Collection<UUID>): List<UUID>

    // AA-I22: photographers the identity notice goes to: not suspended (a closed photographer account is
    // anonymised and suspended) and with no seller record in a passed state.
    @Query(
        """
        SELECT u FROM User u
        WHERE u.role = com.quickpitik.entity.Role.PHOTOGRAPHER AND u.suspendedAt IS NULL
          AND NOT EXISTS (SELECT 1 FROM SellerDetails s WHERE s.userId = u.id AND s.status IN :passed)
          AND (u.identityNoticeSentFor IS NULL OR u.identityNoticeSentFor <> :startsOn)
        ORDER BY u.createdAt ASC, u.id ASC
        """,
    )
    fun findIdentityNoticeRecipients(
        @Param("passed") passed: Collection<SellerDetailsStatus>,
        @Param("startsOn") startsOn: LocalDate,
    ): List<User>

    // Each commits on its own (the notice service has no transaction): a claim is saved before its mail goes out.
    // Returns 1 only for the caller that marked the user, 0 if this start date was already marked.
    @Modifying
    @Transactional
    @Query(
        "UPDATE User u SET u.identityNoticeSentFor = :startsOn WHERE u.id = :id " +
            "AND (u.identityNoticeSentFor IS NULL OR u.identityNoticeSentFor <> :startsOn)",
    )
    fun claimIdentityNotice(@Param("id") id: UUID, @Param("startsOn") startsOn: LocalDate): Int

    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.identityNoticeSentFor = NULL WHERE u.id = :id AND u.identityNoticeSentFor = :startsOn")
    fun clearIdentityNotice(@Param("id") id: UUID, @Param("startsOn") startsOn: LocalDate): Int

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
