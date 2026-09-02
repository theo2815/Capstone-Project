package com.quickpitik.repository

import com.quickpitik.entity.Flag
import org.springframework.data.domain.Page
import org.springframework.data.domain.Pageable
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.util.UUID

interface FlagRepository : JpaRepository<Flag, UUID> {

    @Query(
        """
        SELECT f FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query IS NULL OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        ORDER BY f.createdAt DESC, f.id ASC
        """,
        countQuery = """
        SELECT COUNT(f) FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
          AND (:query IS NULL OR LOWER(f.reason) LIKE LOWER(CONCAT('%', :query, '%')) OR LOWER(f.note) LIKE LOWER(CONCAT('%', :query, '%')))
        """,
    )
    fun pageForAdmin(
        @Param("statusWire") statusWire: String?,
        @Param("query") query: String?,
        pageable: Pageable,
    ): Page<Flag>

    fun countByStatusWire(statusWire: String): Long

    fun existsByTargetKindWireAndTargetIdAndStatusWireAndIdNot(
        targetKindWire: String,
        targetId: UUID,
        statusWire: String,
        id: UUID,
    ): Boolean
}
