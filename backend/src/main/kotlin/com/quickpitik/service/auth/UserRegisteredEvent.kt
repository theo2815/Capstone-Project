package com.quickpitik.service.auth

import java.util.UUID

// Published by AuthService.register. Consumed AFTER_COMMIT so a rolled-back
// registration can never mail a verification link for a user row that does not
// exist — the same reason OrderPaidEvent exists.
data class UserRegisteredEvent(val userId: UUID)
