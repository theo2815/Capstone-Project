package com.quickpitik.repository

import com.quickpitik.entity.ReservedHandle
import org.springframework.data.jpa.repository.JpaRepository

interface ReservedHandleRepository : JpaRepository<ReservedHandle, String> {
    fun existsByHandle(handle: String): Boolean
}
