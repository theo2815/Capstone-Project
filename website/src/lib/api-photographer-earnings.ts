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
//
// Photographer-initiated payout request flow (replaces the prior admin-
// generated weekly cycles):
//   GET  /api/v1/me/photographer/payouts/balance         → PayoutBalanceResponse
//   POST /api/v1/me/photographer/payouts/request         → PhotographerPayout
//   POST /api/v1/me/photographer/payouts/{id}/withdraw   → 204 (no body)

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
  // Pre-fetch up to BE MAX_LIMIT so the Hybrid Load-More UI has rows to chunk
  // through client-side. A bare default of 8 made the slab terminal at "All 8
  // loaded" before the first Load More click.
  p.set("limit", String(args.limit ?? 200));
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
  // Pre-fetch up to BE MAX_LIMIT so the Recent-cycles list has rows to chunk
  // through client-side (PAGE_SIZE.PAYOUT_INCREMENT at a time).
  p.set("limit", String(args.limit ?? 200));
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

// ───────────────────────────────────────────── Payout requests

export interface PayoutBalancePrimaryAccount {
  method: "gcash" | "maya" | "gotyme";
  accountNumber: string;
  accountName: string;
}

export interface PayoutBalanceResponse {
  unpaidBalance: number;
  minimum: number;
  hasOpenRequest: boolean;
  openRequest: PhotographerPayout | null;
  // Backend-authoritative primary payout account (no QR). The request-payout
  // hero + modal gate on this so the FE never believes a primary exists when
  // the DB has none. Null when none is configured.
  primaryAccount: PayoutBalancePrimaryAccount | null;
}

export async function fetchPhotographerPayoutBalance(): Promise<PayoutBalanceResponse | null> {
  return api.get<PayoutBalanceResponse>("/me/photographer/payouts/balance");
}

export async function requestPhotographerPayout(): Promise<PhotographerPayout> {
  return api.post<PhotographerPayout>("/me/photographer/payouts/request");
}

export async function withdrawPhotographerPayout(cycleId: string): Promise<void> {
  await api.post<void>(
    `/me/photographer/payouts/${encodeURIComponent(cycleId)}/withdraw`,
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
  // Pre-fetch up to BE MAX_LIMIT (200) so the Hybrid Load-More UI has rows
  // to chunk through client-side at PAGE_SIZE.TRANSACTION_INCREMENT each
  // click. A bare default of 25 forced the slab to "All 25 loaded" before
  // the first Load More appeared. Photographers with >200 sales would need
  // true infinite-query — out of scope for capstone.
  const limit = args.limit ?? 200;

  const p = new URLSearchParams();
  p.set("offset", String(offset));
  p.set("limit", String(limit));
  return api.get<TransactionsResponse>(
    `/me/photographer/billing/transactions?${p.toString()}`,
  );
}
