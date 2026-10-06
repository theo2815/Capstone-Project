import { api } from "@/lib/api";
import type {
  PhotographerEarnings,
  PhotographerPayout,
  PhotographerTransaction,
  PhotographerLibraryPhoto,
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
//   GET  /api/v1/me/photographer/payouts/reports?cycleId=&status=&offset=&limit= → PaginatedResponse<PayoutReport>
//   GET  /api/v1/me/photographer/billing/transactions?offset=&limit=
//        → PaginatedResponse<PhotographerTransaction> & { monthTotals: Record<string, number> }
//
// Photographer-initiated payout request flow (replaces the prior admin-
// generated weekly cycles):
//   GET  /api/v1/me/photographer/payouts/balance         → PayoutBalanceResponse
//   POST /api/v1/me/photographer/payouts/request         → PhotographerPayout
//   POST /api/v1/me/photographer/payouts/{id}/withdraw   → 204 (no body)

// Backend's MAX_LIMIT for paginated endpoints. These surfaces pre-fetch a
// full page and chunk through it client-side, so this doubles as the ceiling
// on how many rows a photographer can actually reach. Exported so the UI can
// tell "you have exactly 200" apart from "you have more than we fetched".
export const BE_MAX_LIMIT = 200;

// ───────────────────────────────────────────── Earnings overview

export async function fetchPhotographerEarnings(signal?: AbortSignal): Promise<
  PhotographerEarnings | null
> {
  return api.get<PhotographerEarnings>("/me/photographer/earnings", { signal });
}

export interface PerEventEarning {
  eventId: string;
  eventName: string;
  eventDate: string;
  photoCount: number;
  salesCount: number;
  revenueKept: number;
  deletedAt: string | null;
}

export interface PerEventEarningsArgs {
  offset?: number;
  limit?: number;
}

// Returns the whole envelope, not just `items`. The UI needs `total` to tell
// "exactly 200 events" from "200 of 350" — dropping it forced the load-more
// terminal label to hedge with "the 200 most recent".
export async function fetchPerEventEarnings(
  args: PerEventEarningsArgs = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<PerEventEarning>> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  // Pre-fetch up to BE MAX_LIMIT so the Hybrid Load-More UI has rows to chunk
  // through client-side. A bare default of 8 made the slab terminal at "All 8
  // loaded" before the first Load More click.
  p.set("limit", String(args.limit ?? BE_MAX_LIMIT));
  return api.get<PaginatedResponse<PerEventEarning>>(
    `/me/photographer/earnings/per-event?${p.toString()}`,
    { signal },
  );
}

// ───────────────────────────────────────────── Payouts

// Envelope, not just `items` — same reason as fetchPerEventEarnings above.
export async function fetchPhotographerPayouts(
  args: { offset?: number; limit?: number } = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<PhotographerPayout>> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  // Pre-fetch up to BE MAX_LIMIT so the Recent-payouts list has rows to chunk
  // through client-side (PAGE_SIZE.PAYOUT_INCREMENT at a time).
  p.set("limit", String(args.limit ?? BE_MAX_LIMIT));
  return api.get<PaginatedResponse<PhotographerPayout>>(
    `/me/photographer/payouts?${p.toString()}`,
    { signal },
  );
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
): Promise<PayoutReport[]> {
  const p = new URLSearchParams();
  if (args.cycleId) p.set("cycleId", args.cycleId);
  if (args.status) p.set("status", args.status);
  // First page only (backend default 50) — reports are rare per photographer.
  const res = await api.get<PaginatedResponse<PayoutReport>>(
    `/me/photographer/payouts/reports?${p.toString()}`,
  );
  return res.items;
}

// ───────────────────────────────────────────── Payout requests

export interface PayoutBalancePrimaryAccount {
  method: "gcash" | "maya" | "gotyme";
  accountNumber: string;
  accountName: string;
  hasQr: boolean;
}

export interface PayoutBalanceResponse {
  // Withdrawable now = max(0, cleared − paid − reserved). Gate the request CTA
  // on this, never on lifetime earnings.
  unpaidBalance: number;
  // Earned but still inside the clearing window — becomes withdrawable once it
  // ages out. Shown as a separate "clearing" figure so the photographer can see
  // why available < lifetime.
  clearing: number;
  // Tied up in an open request (pending/approved/processing/held); already
  // claimed, so excluded from unpaidBalance.
  reserved: number;
  // ISO timestamp; non-null while a recent payout-account change still bars
  // requests. The hero shows a paused banner and disables Request until it passes.
  accountCooldownUntil: string | null;
  minimum: number;
  hasOpenRequest: boolean;
  openRequest: PhotographerPayout | null;
  // Backend-authoritative primary payout account (no QR). The request-payout
  // hero + modal gate on this so the FE never believes a primary exists when
  // the DB has none. Null when none is configured.
  primaryAccount: PayoutBalancePrimaryAccount | null;
}

export async function fetchPhotographerPayoutBalance(signal?: AbortSignal): Promise<PayoutBalanceResponse | null> {
  return api.get<PayoutBalanceResponse>("/me/photographer/payouts/balance", { signal });
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
  signal?: AbortSignal,
): Promise<TransactionsResponse> {
  const offset = args.offset ?? 0;
  // Pre-fetch up to BE MAX_LIMIT (200) so the Hybrid Load-More UI has rows
  // to chunk through client-side at PAGE_SIZE.TRANSACTION_INCREMENT each
  // click. A bare default of 25 forced the slab to "All 25 loaded" before
  // the first Load More appeared. Photographers with >200 sales would need
  // true infinite-query — out of scope for capstone.
  const limit = args.limit ?? BE_MAX_LIMIT;

  const p = new URLSearchParams();
  p.set("offset", String(offset));
  p.set("limit", String(limit));
  return api.get<TransactionsResponse>(
    `/me/photographer/billing/transactions?${p.toString()}`,
    { signal },
  );
}

export interface PhotographerPurchasedPhoto {
  photo: PhotographerLibraryPhoto;
  purchaseCount: number;
  lastPurchasedAt: string | null;
}

export interface PurchaseEvent {
  id: string;
  name: string;
  date: string;
  deletedAt: string | null;
}

export interface PurchaseTransaction {
  id: string;
  isRefund: boolean;
  refundOf: string | null;
  amountKept: number;
  discountPhp: number;
  paidAt: string;
  recordedAt: string;
}

export interface PhotographerPurchase {
  orderId: string;
  photoId: string;
  buyer: string;
  purchasedAt: string | null;
  orderCreatedAt: string;
  paymentMethod: string;
  pricePhp: number;
  discountPhp: number;
  paidPhp: number;
  kind: string;
  refundStatus: string;
  refundPending: boolean;
  refundedPhp: number | null;
  refundedAt: string | null;
  netKeptPhp: number | null;
  transactions: PurchaseTransaction[];
}

export interface PhotographerPhotoPurchasesResponse extends PaginatedResponse<PhotographerPurchase> {
  photo: PhotographerLibraryPhoto;
  event: PurchaseEvent;
  selectedPurchase: PhotographerPurchase | null;
}

export function fetchPhotographerPurchasedPhotos(
  eventId: string,
  args: { offset?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<PaginatedResponse<PhotographerPurchasedPhoto>> {
  const p = new URLSearchParams({ offset: String(args.offset ?? 0), limit: String(args.limit ?? 60) });
  return api.get(`/me/photographer/events/${encodeURIComponent(eventId)}/purchased-photos?${p}`, { signal: args.signal });
}

export function fetchPhotographerPhotoPurchases(
  photoId: string,
  args: { orderId?: string; offset?: number; limit?: number; signal?: AbortSignal } = {},
): Promise<PhotographerPhotoPurchasesResponse> {
  const p = new URLSearchParams({ offset: String(args.offset ?? 0), limit: String(args.limit ?? 25) });
  if (args.orderId) p.set("orderId", args.orderId);
  return api.get(`/me/photographer/photos/${encodeURIComponent(photoId)}/purchases?${p}`, { signal: args.signal });
}
