package com.quickpitik.repository

import com.quickpitik.entity.RelevanceStrike
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

// RV-2.1: relevance_strikes (V106). RV-4.1 adds the queries.
interface RelevanceStrikeRepository : JpaRepository<RelevanceStrike, UUID>
