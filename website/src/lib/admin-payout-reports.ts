// Photographer-filed payout reports — the return channel from photographer
// to admin. Submitted from /dashboard/billing per cycle row; surfaces in
// the new "Reports from photographers" section on /admin/payouts.
//
// State machine: open → acknowledged → resolved. Each transition pushes a
// PhotographerMessage to the photographer's inbox so the loop closes
// without email. Mirror of the dispute pattern; mock-only until Phase F.

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

// One demo report so the admin section has content on first paint without
// the photographer needing to file one. Tied to a paid cycle (the most
// common report shape: "you marked it paid but money never landed").
export const ADMIN_PAYOUT_REPORTS: ReadonlyArray<PayoutReport> = [
  {
    id: "RPT-SEED-001",
    payoutCycleId: "PAY-2026W14-CEBUSTRIDE",
    photographerId: "photog-cebustride",
    photographerName: "Cebu Stride",
    handle: "cebustride",
    reason: "not_received",
    note:
      "Reference shows MY-32C887 settled Apr 8 but Maya statement has no incoming. Checked twice across both linked numbers.",
    status: "open",
    reportedAt: "2026-04-09T03:42:00.000Z",
    acknowledgedAt: null,
    acknowledgeReply: null,
    resolvedAt: null,
    resolutionNote: null,
  },
];

export function mergeReport(
  base: PayoutReport,
  patch: Partial<PayoutReport> | undefined,
): PayoutReport {
  return patch ? { ...base, ...patch } : base;
}

export function getEffectiveReports(
  submissions: ReadonlyArray<PayoutReport>,
  overrides: Record<string, Partial<PayoutReport>>,
): PayoutReport[] {
  const seed = ADMIN_PAYOUT_REPORTS.map((r) => mergeReport(r, overrides[r.id]));
  const submitted = submissions.map((r) => mergeReport(r, overrides[r.id]));
  return [...submitted, ...seed];
}

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
