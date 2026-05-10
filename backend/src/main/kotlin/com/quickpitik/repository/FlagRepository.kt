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
        ORDER BY f.createdAt DESC, f.id ASC
        """,
        countQuery = """
        SELECT COUNT(f) FROM Flag f
        WHERE (:statusWire IS NULL OR f.statusWire = :statusWire)
        """,
    )
    fun pageByStatus(
        @Param("statusWire") statusWire: String?,
        pageable: Pageable,
    ): Page<Flag>

    fun countByStatusWire(statusWire: String): Long
}
