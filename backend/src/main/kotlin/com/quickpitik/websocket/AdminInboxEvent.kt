package com.quickpitik.websocket

// Published when something happens that an admin needs to see in their
// queue: photographer submits for verification, runner files a dispute,
// photographer files a payout report. Broadcast by NotificationBroadcaster
// on AFTER_COMMIT to every connected admin session.
//
// `payload` is intentionally minimal — admin list views carry hydrated
// joins (decision-log counts, photographer brand, etc.) that would diverge
// if we pushed the full row, so the FE refetches the list it cares about
// using `type` as the dispatch key. Schema:
//   {
//     "type": "verification_submitted" | "dispute_filed" | "payout_report_filed",
//     "entityId": "<uuid|cycleId>",   // userId | disputeId | reportId
//     "actorId":  "<uuid>",            // photographerId | runnerId | photographerId
//     "occurredAt": "<ISO-8601>"
//   }
data class AdminInboxEvent(
    val payload: Map<String, Any?>,
)
