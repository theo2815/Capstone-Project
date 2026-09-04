import { api } from "@/lib/api";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import type {
  PayoutAccount,
  SocialLink,
} from "@/store/photographer-settings-store";
import type {
  Dispute,
  DisputeResolution,
} from "@/lib/admin-disputes";
import type { Flag, FlagStatus } from "@/lib/admin-flags";
import type {
  AdminPayoutCycle,
  AdminPayoutStatus,
} from "@/lib/admin-payouts";
import type {
  PayoutReport,
  PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { ListEvent } from "@/app/events/events-browser";
import type {
  EventReviewStatus,
  PendingPricingChange,
} from "@/lib/photographer-mock";
import type { DecisionLogEntry } from "@/store/admin-user-store";
import type { PaginatedResponse } from "@/types/api";
import { safeUUID } from "@/lib/utils";

// Admin queues fetch a page and then categorize/merge/sort it CLIENT-SIDE
// (status slabs, optimistic-store overrides, GMV sort), so true offset
// pagination would only ever transform the loaded page. They instead pull up
// to the BE max in one fetch and keep their own client-slice Load-more, so the
// rendered DOM stays bounded. Fully scalable admin pagination would need
// server-side categorization — filed as future work.
const ADMIN_LIST_LIMIT = 200;

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
  /** V46 — photographer-owned events awaiting a decision (new + pricing change). */
  pendingEventRequests: number;
}

export interface AdminTrendPoint {
  date: string;
  decisions: number;
  disputes: number;
  payouts: number;
}

export async function fetchAdminKpis(): Promise<AdminKpis> {
  return api.get<AdminKpis>("/admin/kpis");
}

export async function fetchAdminKpiTrend(
  days: number = 30,
): Promise<AdminTrendPoint[]> {
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
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  return p.toString();
}

export async function fetchAdminUsers(
  args: AdminUserListArgs = {},
): Promise<AdminUserRow[]> {
  const res = await api.get<PaginatedResponse<AdminUserRow>>(
    `/admin/users?${buildUsersQs(args)}`,
  );
  return res.items;
}

export async function fetchAdminUserDetail(
  userId: string,
): Promise<AdminUserDetail | null> {
  return api.get<AdminUserDetail>(
    `/admin/users/${encodeURIComponent(userId)}`,
  );
}

// F-NEW-1 — full photographer-settings read for admin review surfaces.
// Returns presigned URLs for cover/watermark/payout-QR so the admin can
// preview the actual media a photographer uploaded.

export interface AdminPhotographerSettingsRegion {
  regionCode: string;
  provinceCode: string;
  city: string | null;
}

export interface AdminPhotographerSettingsCover {
  url: string | null;
  gradientFrom: string | null;
  gradientTo: string | null;
}

export interface AdminPhotographerSettingsWatermark {
  /** Named `dataUrl` to match the FE's WatermarkPreview shape inherited
   *  from the localStorage prototype; with backend storage the value is a
   *  presigned URL. Null when only a label was set. */
  dataUrl: string | null;
  label: string | null;
}

export interface AdminPhotographerSettingsResponse {
  userId: string;
  handle: string | null;
  brandName: string | null;
  brandColor: string;
  bio: string;
  /** Presigned avatar URL via UserDtoMapper.resolveAvatarUrl. Null when
   *  the photographer hasn't uploaded one. */
  avatarUrl: string | null;
  region: AdminPhotographerSettingsRegion | null;
  cover: AdminPhotographerSettingsCover | null;
  watermark: AdminPhotographerSettingsWatermark | null;
  socials: SocialLink[];
  payouts: PayoutAccount[];
}

export async function fetchAdminPhotographerSettings(
  userId: string,
): Promise<AdminPhotographerSettingsResponse> {
  return api.get<AdminPhotographerSettingsResponse>(
    `/admin/users/${encodeURIComponent(userId)}/settings`,
  );
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
  reason: string,
): Promise<AdminUserRow> {
  return api.post<AdminUserRow>(
    `/admin/users/${encodeURIComponent(userId)}/reset-verification`,
    { reason },
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

// Admin → photographer free-form DM. Writes a photographer_messages row
// via AdminDecisionLogService.pushMessage server-side (kind=admin_message,
// title=subject). The photographer reads it via the same /me/photographer/messages
// endpoint as every other admin-action message.
export interface SendAdminMessageInput {
  subject: string;
  body: string;
}

export interface AdminMessageResponse {
  id: string;
  kind: string;
  title: string | null;
  body: string;
  sourceDecisionId: string | null;
  createdAt: string;
  readAt: string | null;
}

export async function sendAdminMessage(
  userId: string,
  input: SendAdminMessageInput,
): Promise<AdminMessageResponse> {
  return api.post<AdminMessageResponse>(
    `/admin/users/${encodeURIComponent(userId)}/message`,
    input,
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
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  return p.toString();
}

export async function fetchAdminDisputes(
  args: AdminDisputeListArgs = {},
): Promise<Dispute[]> {
  const res = await api.get<PaginatedResponse<Dispute>>(
    `/admin/disputes?${buildDisputesQs(args)}`,
  );
  return res.items;
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

// ───────────────────────────────────────────── Flags

export interface AdminFlagListArgs {
  status?: FlagStatus;
  q?: string;
  offset?: number;
  limit?: number;
}

function buildFlagsQs(args: AdminFlagListArgs): string {
  const p = new URLSearchParams();
  if (args.status) p.set("status", args.status);
  if (args.q) p.set("q", args.q);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  return p.toString();
}

export function fetchAdminFlags(
  args: AdminFlagListArgs = {},
): Promise<PaginatedResponse<Flag>> {
  return api.get<PaginatedResponse<Flag>>(
    `/admin/flags?${buildFlagsQs(args)}`,
  );
}

export async function hideAdminFlag(
  flagId: string,
  resolutionNote?: string | null,
): Promise<Flag> {
  return api.post<Flag>(
    `/admin/flags/${encodeURIComponent(flagId)}/hide`,
    { resolutionNote },
  );
}

export async function dismissAdminFlag(
  flagId: string,
  resolutionNote?: string | null,
): Promise<Flag> {
  return api.post<Flag>(
    `/admin/flags/${encodeURIComponent(flagId)}/dismiss`,
    { resolutionNote },
  );
}

export async function escalateAdminFlag(
  flagId: string,
  resolutionNote?: string | null,
): Promise<Flag> {
  return api.post<Flag>(
    `/admin/flags/${encodeURIComponent(flagId)}/escalate`,
    { resolutionNote },
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
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  return p.toString();
}

export async function fetchAdminPayouts(
  args: AdminPayoutListArgs = {},
): Promise<AdminPayoutCycle[]> {
  const res = await api.get<PaginatedResponse<AdminPayoutCycle>>(
    `/admin/payouts?${buildPayoutsQs(args)}`,
  );
  return res.items;
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

// BE requires Idempotency-Key (UUID v4) on POST /admin/payouts/bulk per
// AdminPayoutsController.bulk — retry safety hinges on it. The header is
// generated per call so each batch is a distinct logical decision.
function idempotencyHeader(): { headers: HeadersInit } {
  return { headers: { "Idempotency-Key": safeUUID() } };
}

export async function bulkApprovePayouts(
  payoutIds: string[],
): Promise<BulkPayoutResult> {
  return api.post<BulkPayoutResult>(
    "/admin/payouts/bulk",
    { ids: payoutIds, action: "approve" },
    idempotencyHeader(),
  );
}

export async function bulkHoldPayouts(
  payoutIds: string[],
  reason: string,
): Promise<BulkPayoutResult> {
  return api.post<BulkPayoutResult>(
    "/admin/payouts/bulk",
    { ids: payoutIds, action: "hold", reason },
    idempotencyHeader(),
  );
}

// Admin-triggered cycle generator endpoint still exists on BE
// (POST /admin/payouts/generate) as a safety hatch but the FE no longer
// surfaces it — photographers request payouts via /dashboard/billing in the
// request-based flow.

// ───────────────────────────────────────────── Payout reports

export interface AdminPayoutReportListArgs {
  status?: PayoutReportStatus;
  offset?: number;
  limit?: number;
}

export async function fetchAdminPayoutReports(
  args: AdminPayoutReportListArgs = {},
): Promise<PayoutReport[]> {
  const p = new URLSearchParams();
  if (args.status) p.set("status", args.status);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  const res = await api.get<PaginatedResponse<PayoutReport>>(
    `/admin/payouts/reports?${p.toString()}`,
  );
  return res.items;
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
  /** `queue` (V46) = photographer-owned events awaiting review — a new
   *  submission or a parked pricing change on a live event. */
  review?: "queue";
  offset?: number;
  limit?: number;
}

// AdminListEventDto — the catalog row plus the V46 ownership + review fields.
// createdBy* are null for admin-created events.
export interface AdminEventRow extends ListEvent {
  createdByHandle: string | null;
  createdByName: string | null;
  visibility: "public" | "unlisted";
  pricingMode: "paid" | "free";
  watermarkPolicy: "platform" | "own" | "none";
  reviewStatus: EventReviewStatus;
  reviewNote: string | null;
  pendingChange: PendingPricingChange | null;
}

export async function fetchAdminEvents(
  args: AdminEventListArgs = {},
): Promise<AdminEventRow[]> {
  const p = new URLSearchParams();
  if (args.state) p.set("state", args.state);
  if (args.review) p.set("review", args.review);
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  const res = await api.get<PaginatedResponse<AdminEventRow>>(
    `/admin/events?${p.toString()}`,
  );
  return res.items;
}

// Photographer-owned event review (V46).
//   POST /admin/events/{id}/approve — PENDING → live; CHANGE_PENDING → apply the parked trio
//   POST /admin/events/{id}/reject  — PENDING → rejected (+reason); CHANGE_PENDING → drop the request
export async function approveAdminEvent(
  eventId: string,
): Promise<AdminEventRow> {
  return api.post<AdminEventRow>(
    `/admin/events/${encodeURIComponent(eventId)}/approve`,
    {},
  );
}

export async function rejectAdminEvent(
  eventId: string,
  reason: string,
): Promise<AdminEventRow> {
  return api.post<AdminEventRow>(
    `/admin/events/${encodeURIComponent(eventId)}/reject`,
    { reason },
  );
}

export interface CreateAdminEventArgs {
  title: string;
  date: string;
  location: string;
  /** Per-photo price in PHP. Admin-set at create; the BE seeds new uploads
   *  to this value via PhotoUploadService. Defaults to 0 if omitted. */
  pricePerPhoto: number;
  /** Raw file from the picker. The backend re-encodes to 1920×1440 JPEG
   *  via EventCoverService — sending the raw file avoids the data-URL
   *  detour that overflowed banner_url VARCHAR(512) and 500'd the create. */
  cover?: File | null;
  /** Organizer name + race-day notes for the "About this race" strip. */
  organizerName?: string;
  description?: string;
}

export async function createAdminEvent(
  args: CreateAdminEventArgs,
): Promise<ListEvent> {
  const form = new FormData();
  form.append("title", args.title);
  form.append("date", args.date);
  form.append("location", args.location);
  form.append("pricePerPhoto", String(args.pricePerPhoto));
  if (args.organizerName !== undefined)
    form.append("organizerName", args.organizerName);
  if (args.description !== undefined)
    form.append("description", args.description);
  if (args.cover) form.append("cover", args.cover);
  return api.post<ListEvent>("/admin/events", form);
}

export interface UpdateAdminEventPatch {
  title?: string;
  date?: string;
  location?: string;
  /** New per-photo price in PHP. When this changes the BE re-prices every
   *  existing photo under the event (UPDATE photos SET price_php = ?).
   *  Signed-in carts follow along: `GET /me/cart` renders the live
   *  `photos.price_php`, and checkout charges the same, so there is no drift
   *  to fail on. (An earlier comment here claimed active carts fail with
   *  CART_ITEM_PRICE_CHANGED — they never did; that code only fires on
   *  `CartService.add`.) */
  pricePerPhoto?: number;
  /** New cover file. Wins over `removeCover` when both are present. */
  cover?: File | null;
  /** Clear the existing cover key on the server. Ignored if `cover` is set. */
  removeCover?: boolean;
  /** New organizer name / race-day notes. Only forwarded when changed in the
   *  modal; blank is a no-op on the backend (same contract as title/location). */
  organizerName?: string;
  description?: string;
}

export async function updateAdminEvent(
  eventId: string,
  patch: UpdateAdminEventPatch,
): Promise<ListEvent> {
  const form = new FormData();
  if (patch.title !== undefined) form.append("title", patch.title);
  if (patch.date !== undefined) form.append("date", patch.date);
  if (patch.location !== undefined) form.append("location", patch.location);
  if (patch.pricePerPhoto !== undefined) {
    form.append("pricePerPhoto", String(patch.pricePerPhoto));
  }
  if (patch.organizerName !== undefined)
    form.append("organizerName", patch.organizerName);
  if (patch.description !== undefined)
    form.append("description", patch.description);
  if (patch.cover) form.append("cover", patch.cover);
  if (patch.removeCover) form.append("removeCover", "true");
  return api.fetch<ListEvent>(
    `/admin/events/${encodeURIComponent(eventId)}`,
    { method: "PATCH", body: form },
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
): Promise<AdminSalesKpis> {
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
): Promise<AdminSalesEventRow[]> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? ADMIN_LIST_LIMIT));
  if (args.order) p.set("order", args.order);
  const res = await api.get<PaginatedResponse<AdminSalesEventRow>>(
    `/admin/sales/by-event?${p.toString()}`,
  );
  return res.items;
}
