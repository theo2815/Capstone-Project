package com.quickpitik.service.photographer

import java.util.UUID

// Published by PayoutAccountService on every payout-destination change; consumed
// AFTER_COMMIT + async by PayoutAccountChangedEmailListener so the account owner
// is alerted at their registered email even when a compromised session made the
// change — during the 24h cooldown PayoutRequestService enforces. Carries only a
// MASKED account (last 4), never the raw number.
data class PayoutAccountChangedEvent(
    val userId: UUID,
    val action: String,
    val method: String?,
    val maskedAccount: String?,
)
