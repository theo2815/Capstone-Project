import { api } from "@/lib/api";
import type {
  PhotographerEarnings,
  PhotographerPayout,
  PhotographerTransaction,
} from "@/lib/photographer-mock";
import type {
  PayoutReport,
  PayoutReportReason,
  PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { PaginatedResponse } from "@/types/api";

// Phase F.2 photographer-earnings backend contract
//   Q-A1 (cycle ID format), Q-E1 (payout method enum), Q-E2 (QR static),
//   Q-E3 (refund display deferred) RESOLVED 2026-05-10. See vault decisions.
//
//   GET  /api/v1/me/photographer/earnings                                → PhotographerEarnings
//   GET  /api/v1/me/photographer/earnings/per-event?offset=&limit=
//        → PaginatedResponse<PerEventEarning>
//   GET  /api/v1/me/photographer/payouts?offset=&limit=                  → PaginatedResponse<PhotographerPayout>
//   POST /api/v1/me/photographer/payouts/{id}/report { reason, note? }   → PayoutReport
//   GET  /api/v1/me/photographer/payouts/reports?cycleId=&status=        → PayoutReport[]
//   GET  /api/v1/me/photographer/billing/transactions?offset=&limit=
//        → PaginatedResponse<PhotographerTransaction> & { monthTotals: Record<string, number> }

// ───────────────────────────────────────────── Earnings overview

export async function fetchPhotographerEarnings(): Promise<
  PhotographerEarnings | null
> {
  return api.get<PhotographerEarnings>("/me/photographer/earnings");
}

export interface PerEventEarning {
  eventId: string;
  eventName: string;
  eventDate: string;
  photoCount: number;
  salesCount: number;
  revenueKept: number;
}

export interface PerEventEarningsArgs {
  offset?: number;
  limit?: number;
}

export async function fetchPerEventEarnings(
  args: PerEventEarningsArgs = {},
): Promise<PerEventEarning[]> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 8));
  const res = await api.get<PaginatedResponse<PerEventEarning>>(
    `/me/photographer/earnings/per-event?${p.toString()}`,
  );
  return res.items;
}

// ───────────────────────────────────────────── Payouts

export async function fetchPhotographerPayouts(
  args: { offset?: number; limit?: number } = {},
): Promise<PhotographerPayout[]> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  const res = await api.get<PaginatedResponse<PhotographerPayout>>(
    `/me/photographer/payouts?${p.toString()}`,
  );
  return res.items;
}

export interface SubmitPayoutReportArgs {
  payoutId: string;
  reason: PayoutReportReason;
  note: string | null;
}

export async function submitPayoutReport(
  args: SubmitPayoutReportArgs,
): Promise<PayoutReport> {
  return api.post<PayoutReport>(
    `/me/photographer/payouts/${encodeURIComponent(args.payoutId)}/report`,
    { reason: args.reason, note: args.note },
  );
}

export interface PhotographerPayoutReportsArgs {
  cycleId?: string;
  status?: PayoutReportStatus;
}

export async function fetchPhotographerPayoutReports(
  args: PhotographerPayoutReportsArgs = {},
): Promise<PayoutReport[] | null> {
  const p = new URLSearchParams();
  if (args.cycleId) p.set("cycleId", args.cycleId);
  if (args.status) p.set("status", args.status);
  return api.get<PayoutReport[]>(
    `/me/photographer/payouts/reports?${p.toString()}`,
  );
}

// ───────────────────────────────────────────── Transactions ledger

export interface TransactionsResponse {
  items: PhotographerTransaction[];
  total: number;
  offset: number;
  limit: number;
  monthTotals: Record<string, number>;
}

export async function fetchPhotographerTransactions(
  args: { offset?: number; limit?: number } = {},
): Promise<TransactionsResponse> {
  const offset = args.offset ?? 0;
  const limit = args.limit ?? 25;

  const p = new URLSearchParams();
  p.set("offset", String(offset));
  p.set("limit", String(limit));
  return api.get<TransactionsResponse>(
    `/me/photographer/billing/transactions?${p.toString()}`,
  );
}
