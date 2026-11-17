package com.quickpitik.repository

import com.quickpitik.entity.RacePriceChange
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface RacePriceChangeRepository : JpaRepository<RacePriceChange, UUID>
