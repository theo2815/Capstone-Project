package com.quickpitik.service.disbursement

import org.springframework.stereotype.Component

/**
 * Today's rail: the admin opens their GCash/Maya/GoTyme app, sends the money by
 * hand, and records the reference via mark-paid. There is no programmatic
 * transfer to create, so initiate is a no-op that records no provider id — the
 * cycle simply moves to PROCESSING and waits for the admin's attestation.
 */
@Component
class ManualDisbursementProvider : DisbursementProvider {
    override val providerName = "manual"

    override fun initiate(request: DisbursementRequest): DisbursementResult =
        DisbursementResult(providerTransferId = null)
}
