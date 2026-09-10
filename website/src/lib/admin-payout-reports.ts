// Photographer-filed payout reports — the return channel from photographer
// to admin. Submitted from /dashboard/billing per cycle row; surfaces in
// the new "Reports from photographers" section on /admin/payouts.
//
// State machine: open → acknowledged → resolved. Each transition pushes a
// PhotographerMessage to the photographer's inbox so the loop closes
// without email. Records and timestamps are owned by the backend.

export type PayoutReportReason =
  | "not_received"
  | "wrong_amount"
  | "account_info"
  | "processing_delay"
  | "other";

export type PayoutReportStatus = "open" | "acknowledged" | "resolved";

export interface PayoutReport {
  id: string;
  payoutCycleId: string;
  photographerId: string;
  photographerName: string;
  handle: string | null;
  reason: PayoutReportReason;
  note: string;
  status: PayoutReportStatus;
  reportedAt: string;
  acknowledgedAt: string | null;
  acknowledgeReply: string | null;
  resolvedAt: string | null;
  resolutionNote: string | null;
}

export const PAYOUT_REPORT_REASON_LABEL: Record<PayoutReportReason, string> = {
  not_received: "Payout not received",
  wrong_amount: "Wrong amount transferred",
  account_info: "Account info needs update",
  processing_delay: "Processing delay",
  other: "Other",
};

export const PAYOUT_REPORT_STATUS_LABEL: Record<PayoutReportStatus, string> = {
  open: "Open",
  acknowledged: "Acknowledged",
  resolved: "Resolved",
};

// Lookup used by the photographer billing page to decide between
// "File a report" (no prior report) and "Track your report" (any prior
// report). Returns the most recent report by reportedAt — covers the
// follow-up case where a photographer files multiple reports against the
// same cycle, the latest one drives the row's button state.
export function getLatestReportForCycle(
  reports: ReadonlyArray<PayoutReport>,
  photographerId: string,
  payoutCycleId: string,
): PayoutReport | null {
  let latest: PayoutReport | null = null;
  for (const r of reports) {
    if (r.photographerId !== photographerId) continue;
    if (r.payoutCycleId !== payoutCycleId) continue;
    if (!latest || r.reportedAt.localeCompare(latest.reportedAt) > 0) {
      latest = r;
    }
  }
  return latest;
}
