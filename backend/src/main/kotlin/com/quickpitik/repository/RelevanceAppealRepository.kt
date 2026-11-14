package com.quickpitik.repository

import com.quickpitik.entity.RelevanceAppeal
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

// RV-2.1: relevance_appeals (V106). RV-3.2 adds the queries.
interface RelevanceAppealRepository : JpaRepository<RelevanceAppeal, UUID>
