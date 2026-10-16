package com.quickpitik.repository

import com.quickpitik.entity.PayoutCycle

/** Insert-only write for payout cycles: an existing id fails instead of being overwritten. */
interface PayoutCycleInsert {
    fun insert(cycle: PayoutCycle): PayoutCycle
}
