package com.quickpitik.common

import java.math.BigDecimal

// The one peso -> centavo conversion: PayMongo takes amounts in centavos and
// every money column is NUMERIC(_, 2), so a persisted amount always converts
// exactly. longValueExact throws on a fraction of a centavo instead of
// truncating it silently (2026-09-12 — the old buildLineItems copy truncated).
fun BigDecimal.toCentavos(): Long = movePointRight(2).longValueExact()
