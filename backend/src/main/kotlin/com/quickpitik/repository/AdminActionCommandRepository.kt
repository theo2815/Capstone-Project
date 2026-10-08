package com.quickpitik.repository

import com.quickpitik.entity.AdminActionCommand
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface AdminActionCommandRepository : JpaRepository<AdminActionCommand, UUID>
