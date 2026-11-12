package com.quickpitik.service.admin

import java.util.UUID

// PE-2.2: published by AdminFlagService.confirmPayoutHold inside the hold's transaction.
data class PayoutHoldPlacedEvent(val photographerId: UUID)
