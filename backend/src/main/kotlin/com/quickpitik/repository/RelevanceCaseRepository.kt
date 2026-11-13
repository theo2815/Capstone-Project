package com.quickpitik.repository

import com.quickpitik.entity.RelevanceCase
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

// RV-1.1: relevance_cases, written only by service/relevance. Queries come with the tasks that need them.
interface RelevanceCaseRepository : JpaRepository<RelevanceCase, UUID>
