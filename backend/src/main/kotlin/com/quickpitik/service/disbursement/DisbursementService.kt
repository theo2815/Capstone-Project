package com.quickpitik.service.disbursement

import com.quickpitik.config.PlatformProperties
import org.springframework.stereotype.Service

/**
 * Resolves and delegates to the disbursement provider named by
 * `app.platform.disbursement-provider`. All registered providers are injected;
 * the active one is picked by name at call time, so flipping the config (once
 * PayMongo KYB is approved) switches the rail with no code change.
 */
@Service
class DisbursementService(
    providers: List<DisbursementProvider>,
    private val platformProperties: PlatformProperties,
) {
    private val byName: Map<String, DisbursementProvider> =
        providers.associateBy { it.providerName }

    val activeName: String
        get() = platformProperties.disbursementProvider

    fun initiate(request: DisbursementRequest): DisbursementResult =
        active().initiate(request)

    private fun active(): DisbursementProvider =
        byName[activeName]
            ?: throw IllegalStateException(
                "No DisbursementProvider named '$activeName' (registered: ${byName.keys})",
            )
}
