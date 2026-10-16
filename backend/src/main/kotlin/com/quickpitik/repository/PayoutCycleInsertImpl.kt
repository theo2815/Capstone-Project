package com.quickpitik.repository

import com.quickpitik.entity.PayoutCycle
import jakarta.persistence.EntityManager
import org.springframework.transaction.annotation.Transactional

@Transactional
class PayoutCycleInsertImpl(private val entityManager: EntityManager) : PayoutCycleInsert {
    // JpaRepository.save merges an entity whose id the caller assigned: on an id
    // that already exists it overwrites that row (security audit 2026-10-14).
    // persist always INSERTs, so a clash fails on payout_cycles_pkey instead.
    override fun insert(cycle: PayoutCycle): PayoutCycle {
        entityManager.persist(cycle)
        entityManager.flush()
        return cycle
    }
}
