package com.quickpitik.repository

import com.quickpitik.entity.AdminMfa
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface AdminMfaRepository : JpaRepository<AdminMfa, UUID>
