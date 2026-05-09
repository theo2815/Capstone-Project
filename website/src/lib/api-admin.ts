import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { ADMIN_USER_SEED, type AdminUserRow } from "@/lib/admin-user-registry";
import {
  ADMIN_DISPUTES,
  type Dispute,
  type DisputeResolution,
} from "@/lib/admin-disputes";
import {
  ADMIN_PAYOUT_SEED,
  type AdminPayoutCycle,
  type AdminPayoutStatus,
} from "@/lib/admin-payouts";
import {
  ADMIN_PAYOUT_REPORTS,
  type PayoutReport,
  type PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { ListEvent } from "@/app/events/events-browser";
import type { DecisionLogEntry } from "@/store/admin-user-store";
import type { PaginatedResponse } from "@/types/api";

// Phase G admin backend contract (Q-A1 + Q-A2 + Q-A3 + Q-A4 RESOLVED 2026-05-09).
//
//   GET    /api/v1/admin/kpis                              → AdminKpis
//   GET    /api/v1/admin/kpis/trend?days=N                 → AdminTrendPoint[]
//
//   GET    /api/v1/admin/users?role=&status=&q=&offset=&limit=  → PaginatedResponse<AdminUserRow>
//   GET    /api/v1/admin/users/{id}                        → AdminUserDetail
//   POST   /api/v1/admin/users/{id}/approve                → AdminUserRow
//   POST   /api/v1/admin/users/{id}/reject       { reason } → AdminUserRow
//   POST   /api/v1/admin/users/{id}/reset-verification     → AdminUserRow
//   POST   /api/v1/admin/users/{id}/suspend      { reason } → AdminUserRow
//   POST   /api/v1/admin/users/{id}/unsuspend              → AdminUserRow
//   PATCH  /api/v1/admin/users/{id}              { handle?, brandName? } → AdminUserRow
//
//   GET    /api/v1/admin/disputes?status=&offset=&limit=   → PaginatedResponse<Dispute>
//   POST   /api/v1/admin/disputes/{id}/resolve   { resolution, refundAmount?, reason } → Dispute
//   POST   /api/v1/admin/disputes/{id}/deny      { reason } → Dispute
//   POST   /api/v1/admin/disputes/{id}/escalate  { reason } → Dispute
//
//   GET    /api/v1/admin/payouts?status=&offset=&limit=    → PaginatedResponse<AdminPayoutCycle>
//   POST   /api/v1/admin/payouts/{id}/approve              → AdminPayoutCycle
//   POST   /api/v1/admin/payouts/{id}/hold       { reason } → AdminPayoutCycle
//   POST   /api/v1/admin/payouts/{id}/mark-paid  { paymentReference } → AdminPayoutCycle
//   POST   /api/v1/admin/payouts/bulk            { ids, action, reason? } → BulkPayoutResult
//
//   GET    /api/v1/admin/payouts/reports?status=&offset=&limit= → PaginatedResponse<PayoutReport>
//   PATCH  /api/v1/admin/payouts/reports/{id}/acknowledge { reply } → PayoutReport
//   PATCH  /api/v1/admin/payouts/reports/{id}/resolve     { resolutionNote } → PayoutReport
//
//   GET    /api/v1/admin/events?state=&offset=&limit=      → PaginatedResponse<ListEvent>
//   POST   /api/v1/admin/events                  { title, date, location, bannerUrl? } → ListEvent
//   PATCH  /api/v1/admin/events/{id}             { title?, date?, location? } → ListEvent
//   DELETE /api/v1/admin/events/{id}                       → { removed: boolean }
//
//   GET    /api/v1/admin/sales/kpis?range=week|month|ytd   → AdminSalesKpis
//   GET    /api/v1/admin/sales/by-event?offset=&limit=&order=gmv|refunds → PaginatedResponse<AdminSalesEventRow>

// ───────────────────────────────────────────── KPIs

export interface AdminKpis {
  pendingVerifications: number;
  approvedPhotographers: number;
  suspended: number;
  liveEvents: number;
  decisionsThisWeek: number;
  openDisputes: number;
  openFlags: number;
  pendingPayouts: number;
}

export interface AdminTrendPoint {
  date: string;
  decisions: number;
  disputes: number;
  payouts: number;
}

export async function fetchAdminKpis(): Promise<AdminKpis | null> {
  if (!BACKEND_LIVE) return null;
  return api.get<AdminKpis>("/admin/kpis");
}

export async function fetchAdminKpiTrend(
  days: number = 30,
): Promise<AdminTrendPoint[] | null> {
  if (!BACKEND_LIVE) return null;
  return api.get<AdminTrendPoint[]>(`/admin/kpis/trend?days=${days}`);
}

// ───────────────────────────────────────────── Users (verifications + photographers)

export type AdminUserStatusFilter =
  | "pending"
  | "approved"
  | "incomplete"
  | "suspended";

export interface AdminUserListArgs {
  role?: "PHOTOGRAPHER" | "RUNNER";
  status?: AdminUserStatusFilter;
  q?: string;
  offset?: number;
  limit?: number;
}

export interface AdminUserDetail extends AdminUserRow {
  decisionLog: DecisionLogEntry[];
}

function buildUsersQs(args: AdminUserListArgs): string {
  const p = new URLSearchParams();
  if (args.role) p.set("role", args.role);
  if (args.status) p.set("status", args.status);
  if (args.q) p.set("q", args.q);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  return p.toString();
}

export async function fetchAdminUsers(
  args: AdminUserListArgs = {},
): Promise<AdminUserRow[]> {
  if (BACKEND_LIVE) {
    const res = await api.get<PaginatedResponse<AdminUserRow>>(
      `/admin/users?${buildUsersQs(args)}`,
    );
    return res.items;
  }
  return ADMIN_USER_SEED.slice();
}

export async function fetchAdminUserDetail(
  userId: string,
): Promise<AdminUserDetail | null> {
  if (BACKEND_LIVE) {
    return api.get<AdminUserDetail>(
      `/admin/users/${encodeURIComponent(userId)}`,
    );
  }
  const row = ADMIN_USER_SEED.find((u) => u.userId === userId);
  if (!row) return null;
  return { ...row, decisionLog: [] };
}

export async function approveUser(userId: string): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/approve`,
  );
}

export async function rejectUser(
  userId: string,
  reason: string,
): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/reject`,
    { reason },
  );
}

export async function resetUserVerification(
  userId: string,
): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/reset-verification`,
  );
}

export async function suspendUser(
  userId: string,
  reason: string,
): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/suspend`,
    { reason },
  );
}

export async function unsuspendUser(userId: string): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/unsuspend`,
  );
}

export interface ForceEditUserPatch {
  handle?: string | null;
  brandName?: string | null;
}

export async function forceEditUser(
  userId: string,
  patch: ForceEditUserPatch,
): Promise<AdminUserRow> {
  return api.fetch<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}`,
    { method: "PATCH", body: JSON.stringify(patch) },
  );
}

// ───────────────────────────────────────────── Disputes

export type AdminDisputeStatusFilter =
  | "open"
  | "resolved"
  | "denied"
  | "escalated";

export interface AdminDisputeListArgs {
  status?: AdminDisputeStatusFilter;
  q?: string;
  offset?: number;
  limit?: number;
}

function buildDisputesQs(args: AdminDisputeListArgs): string {
  const p = new URLSearchParams();
  if (args.status) p.set("status", args.status);
  if (args.q) p.set("q", args.q);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  return p.toString();
}

export async function fetchAdminDisputes(
  args: AdminDisputeListArgs = {},
): Promise<Dispute[]> {
  if (BACKEND_LIVE) {
    const res = await api.get<PaginatedResponse<Dispute>>(
      `/admin/disputes?${buildDisputesQs(args)}`,
    );
    return res.items;
  }
  return ADMIN_DISPUTES.slice();
}

export interface ResolveDisputeArgs {
  resolution: DisputeResolution;
  refundAmount: number | null;
  reason: string | null;
}

export async function resolveDispute(
  disputeId: string,
  args: ResolveDisputeArgs,
): Promise<Dispute> {
  return api.post<Dispute>(
    `/admin/disputes/${encodeURIComponent(disputeId)}/resolve`,
    args,
  );
}

export async function denyDispute(
  disputeId: string,
  reason: string | null,
): Promise<Dispute> {
  return api.post<Dispute>(
    `/admin/disputes/${encodeURIComponent(disputeId)}/deny`,
    { reason },
  );
}

export async function escalateDispute(
  disputeId: string,
  reason: string | null,
): Promise<Dispute> {
  return api.post<Dispute>(
    `/admin/disputes/${encodeURIComponent(disputeId)}/escalate`,
    { reason },
  );
}

// ───────────────────────────────────────────── Payouts

export interface AdminPayoutListArgs {
  status?: AdminPayoutStatus;
  q?: string;
  offset?: number;
  limit?: number;
}

function buildPayoutsQs(args: AdminPayoutListArgs): string {
  const p = new URLSearchParams();
  if (args.status) p.set("status", args.status);
  if (args.q) p.set("q", args.q);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  return p.toString();
}

export async function fetchAdminPayouts(
  args: AdminPayoutListArgs = {},
): Promise<AdminPayoutCycle[]> {
  if (BACKEND_LIVE) {
    const res = await api.get<PaginatedResponse<AdminPayoutCycle>>(
      `/admin/payouts?${buildPayoutsQs(args)}`,
    );
    return res.items;
  }
  return ADMIN_PAYOUT_SEED.slice();
}

export async function approvePayout(
  payoutId: string,
): Promise<AdminPayoutCycle> {
  return api.post<AdminPayoutCycle>(
    `/admin/payouts/${encodeURIComponent(payoutId)}/approve`,
  );
}

export async function holdPayout(
  payoutId: string,
  reason: string,
): Promise<AdminPayoutCycle> {
  return api.post<AdminPayoutCycle>(
    `/admin/payouts/${encodeURIComponent(payoutId)}/hold`,
    { reason },
  );
}

export async function markPayoutPaid(
  payoutId: string,
  paymentReference: string,
): Promise<AdminPayoutCycle> {
  return api.post<AdminPayoutCycle>(
    `/admin/payouts/${encodeURIComponent(payoutId)}/mark-paid`,
    { paymentReference },
  );
}

export interface BulkPayoutResult {
  groupId: string;
  results: Array<{ id: string; ok: boolean; error?: string }>;
}

export async function bulkApprovePayouts(
  payoutIds: string[],
): Promise<BulkPayoutResult> {
  return api.post<BulkPayoutResult>("/admin/payouts/bulk", {
    ids: payoutIds,
    action: "approve",
  });
}

export async function bulkHoldPayouts(
  payoutIds: string[],
  reason: string,
): Promise<BulkPayoutResult> {
  return api.post<BulkPayoutResult>("/admin/payouts/bulk", {
    ids: payoutIds,
    action: "hold",
    reason,
  });
}

// ───────────────────────────────────────────── Payout reports

export interface AdminPayoutReportListArgs {
  status?: PayoutReportStatus;
  offset?: number;
  limit?: number;
}

export async function fetchAdminPayoutReports(
  args: AdminPayoutReportListArgs = {},
): Promise<PayoutReport[]> {
  if (BACKEND_LIVE) {
    const p = new URLSearchParams();
    if (args.status) p.set("status", args.status);
    p.set("offset", String(args.offset ?? 0));
    p.set("limit", String(args.limit ?? 50));
    const res = await api.get<PaginatedResponse<PayoutReport>>(
      `/admin/payouts/reports?${p.toString()}`,
    );
    return res.items;
  }
  return ADMIN_PAYOUT_REPORTS.slice();
}

export async function acknowledgePayoutReport(
  reportId: string,
  reply: string,
): Promise<PayoutReport> {
  return api.fetch<PayoutReport>(
    `/admin/payouts/reports/${encodeURIComponent(reportId)}/acknowledge`,
    { method: "PATCH", body: JSON.stringify({ reply }) },
  );
}

export async function resolvePayoutReport(
  reportId: string,
  resolutionNote: string,
): Promise<PayoutReport> {
  return api.fetch<PayoutReport>(
    `/admin/payouts/reports/${encodeURIComponent(reportId)}/resolve`,
    { method: "PATCH", body: JSON.stringify({ resolutionNote }) },
  );
}

// ───────────────────────────────────────────── Events catalog

export interface AdminEventListArgs {
  state?: ListEvent["state"];
  offset?: number;
  limit?: number;
}

export async function fetchAdminEvents(
  args: AdminEventListArgs = {},
): Promise<ListEvent[]> {
  if (!BACKEND_LIVE) return [];
  const p = new URLSearchParams();
  if (args.state) p.set("state", args.state);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  const res = await api.get<PaginatedResponse<ListEvent>>(
    `/admin/events?${p.toString()}`,
  );
  return res.items;
}

export interface CreateAdminEventArgs {
  title: string;
  date: string;
  location: string;
  bannerUrl?: string;
}

export async function createAdminEvent(
  args: CreateAdminEventArgs,
): Promise<ListEvent> {
  return api.post<ListEvent>("/admin/events", args);
}

export interface UpdateAdminEventPatch {
  title?: string;
  date?: string;
  location?: string;
}

export async function updateAdminEvent(
  eventId: string,
  patch: UpdateAdminEventPatch,
): Promise<ListEvent> {
  return api.fetch<ListEvent>(
    `/admin/events/${encodeURIComponent(eventId)}`,
    { method: "PATCH", body: JSON.stringify(patch) },
  );
}

export async function deleteAdminEvent(
  eventId: string,
): Promise<{ removed: boolean }> {
  return api.delete<{ removed: boolean }>(
    `/admin/events/${encodeURIComponent(eventId)}`,
  );
}

// ───────────────────────────────────────────── Sales

export type AdminSalesRange = "week" | "month" | "ytd";

export interface AdminSalesKpis {
  gmv: number;
  platformRevenue: number;
  refundsIssued: number;
  netPlatformRevenue: number;
  photographerKeep: number;
  totalSalesCount: number;
}

export async function fetchAdminSalesKpis(
  range: AdminSalesRange = "ytd",
): Promise<AdminSalesKpis | null> {
  if (!BACKEND_LIVE) return null;
  return api.get<AdminSalesKpis>(`/admin/sales/kpis?range=${range}`);
}

export interface AdminSalesEventRow {
  id: string;
  slug: string;
  name: string;
  date: string;
  city: string;
  status: string;
  state: string;
  photoCount: number;
  impliedGmv: number;
  impliedCut: number;
  refundsIssued: number;
}

export interface AdminSalesByEventArgs {
  offset?: number;
  limit?: number;
  order?: "gmv" | "refunds";
}

export async function fetchAdminSalesByEvent(
  args: AdminSalesByEventArgs = {},
): Promise<AdminSalesEventRow[] | null> {
  if (!BACKEND_LIVE) return null;
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 50));
  if (args.order) p.set("order", args.order);
  const res = await api.get<PaginatedResponse<AdminSalesEventRow>>(
    `/admin/sales/by-event?${p.toString()}`,
  );
  return res.items;
}
