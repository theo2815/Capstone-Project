import { api } from "@/lib/api";
import { API_BASE_URL, ROUTES } from "@/lib/constants";
import type {
  DisputeReason,
  DisputeResolution,
} from "@/lib/admin-disputes";
import type { MockOrder } from "@/store/orders-store";
import type { PaginatedResponse } from "@/types/api";
import type { Order, OrderPhotoDetail, OrderStatus } from "@/types/order";

// Embedded on every MockOrder / OrderDetail payload from /me/orders. Carries
// what the /orders surface needs to render the status chip, timeline, cancel
// button, and admin's resolution note — without a separate per-order fetch.
// Mirrors the BE RunnerDisputeDto.
export type RunnerDisputeStatus =
  | "open"
  | "resolved"
  | "denied"
  | "escalated"
  | "withdrawn";

export interface RunnerDispute {
  id: string;
  photoId: string;
  reason: DisputeReason;
  note: string;
  status: RunnerDisputeStatus;
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  // Most-recent admin_decision_log.reason for this dispute (resolve / deny /
  // escalate). Null when admin hasn't engaged yet.
  resolutionNote: string | null;
  openedAt: string;
  resolvedAt: string | null;
  withdrawnAt: string | null;
}

// Orders backend contract (Q-008 idempotency, Q-005 photo URLs RESOLVED 2026-05-09).
//   GET  /api/v1/me/orders?offset=&limit=    → PaginatedResponse<MockOrder>
//   GET  /api/v1/me/orders/{id}              → OrderDetail (MockOrder + photos[] + downloadBundleUrl)
//   POST /api/v1/orders                      { items, paymentMethod, recipientEmail?, couponCode? } + Idempotency-Key header → Order
//   POST /api/v1/me/orders/{id}/refund       { photoIds, reason, note } → unknown
//
// Idempotency-Key arrives as an HTTP header per RFC 9110 §9.2.2.

export interface OrderDetail extends MockOrder {
  photos: OrderPhotoDetail[];
  downloadBundleUrl?: string;
  // Populated for the /orders/return success state. May be empty on legacy
  // payloads or when the BE chose to omit it.
  recipientEmail?: string;
  // Per-order opaque token. Both runners and guests use it to authorize the
  // bundle-download URL (a top-level <a> navigation can't carry the JWT).
  // Null on legacy rows; FE hides the master Download-all button when absent.
  shareToken?: string | null;
}

export interface CreateOrderArgs {
  items: { photoId: string; eventId: string }[];
  paymentMethod: "qrph";
  recipientEmail?: string;
  // Photographer coupon (V45). Validated and priced server-side; the client
  // only forwards what the runner typed.
  couponCode?: string;
  idempotencyKey: string;
}

export interface RefundSubmitArgs {
  orderId: string;
  photoIds: string[];
  reason: string;
  note: string;
}

export async function fetchOrders(
  args: { offset?: number; limit?: number } = {},
): Promise<MockOrder[]> {
  const offset = args.offset ?? 0;
  const limit = args.limit ?? 200;

  const params = new URLSearchParams();
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  const res = await api.get<PaginatedResponse<MockOrder>>(
    `/me/orders?${params.toString()}`,
  );
  return res.items;
}

export async function fetchOrderDetail(
  orderId: string,
): Promise<OrderDetail | null> {
  return api.get<OrderDetail>(`/me/orders/${encodeURIComponent(orderId)}`);
}

export async function postOrder(args: CreateOrderArgs): Promise<Order> {
  const { idempotencyKey, ...body } = args;
  return api.post<Order>("/orders", body, {
    headers: { "Idempotency-Key": idempotencyKey },
  });
}

export async function submitOrderRefund(
  args: RefundSubmitArgs,
): Promise<unknown> {
  return api.post<unknown>(
    `/me/orders/${encodeURIComponent(args.orderId)}/refund`,
    {
      photoIds: args.photoIds,
      reason: args.reason,
      note: args.note,
    },
  );
}

// Runner cancels their own open refund request. BE returns the updated
// RunnerDispute (status flipped to "withdrawn", withdrawnAt stamped). Caller
// should invalidate the orders cache so the receipt chip + photo lock clear.
export async function withdrawDispute(
  disputeId: string,
): Promise<RunnerDispute> {
  return api.post<RunnerDispute>(
    `/me/disputes/${encodeURIComponent(disputeId)}/withdraw`,
    {},
  );
}

// Guest-friendly status poll. The /orders/return page calls this to check
// whether PayMongo's webhook has fired yet. Authed runners hit
// `fetchOrderDetail` instead — they get full hydration via JWT.
export interface OrderStatusPayload {
  id: string;
  status: OrderStatus;
  paidAt: string | null;
}

// `verify: true` makes the backend ask PayMongo about the pending intent right
// now (and settle it) instead of waiting for the webhook — used when the
// runner says "I've paid". Rate-limited tighter than the plain read, so the
// automatic poll never sets it.
export interface OrderStatusOptions {
  verify?: boolean;
}

export async function fetchOrderStatus(
  orderId: string,
  token: string,
  opts: OrderStatusOptions = {},
): Promise<OrderStatusPayload> {
  const params = new URLSearchParams({ token });
  if (opts.verify) params.set("verify", "true");
  return api.get<OrderStatusPayload>(
    `/orders/${encodeURIComponent(orderId)}/status?${params.toString()}`,
  );
}

export async function fetchOrderStatusForUser(
  orderId: string,
  opts: OrderStatusOptions = {},
): Promise<OrderStatusPayload> {
  const qs = opts.verify ? "?verify=true" : "";
  return api.get<OrderStatusPayload>(
    `/me/orders/${encodeURIComponent(orderId)}/status${qs}`,
  );
}

// One status call for a pending record: guest orders use their signed return
// token, signed-in orders the JWT route. Shared by the checkout modal and the
// background watcher so both pick the endpoint the same way.
export function fetchPendingStatus(
  pending: { orderId: string; returnToken: string | null },
  opts: OrderStatusOptions = {},
): Promise<OrderStatusPayload> {
  return pending.returnToken
    ? fetchOrderStatus(pending.orderId, pending.returnToken, opts)
    : fetchOrderStatusForUser(pending.orderId, opts);
}

// Runner backs out of a live QR. The backend asks PayMongo once more first,
// so the returned status is the truth: PAID/FULFILLED means the payment won
// the race and the caller should show success, not "cancelled".
export function cancelPendingPayment(
  pending: { orderId: string; returnToken: string | null },
): Promise<OrderStatusPayload> {
  const id = encodeURIComponent(pending.orderId);
  return pending.returnToken
    ? api.post<OrderStatusPayload>(
        `/orders/${id}/cancel?${new URLSearchParams({ token: pending.returnToken })}`,
      )
    : api.post<OrderStatusPayload>(`/me/orders/${id}/cancel`);
}

// The backend records a declined payment and a timed-out QR the same way
// (EXPIRED). Seen before the QR's own deadline it can only be a failure.
export function classifyExpired(
  expiresAt: string,
  now: number = Date.now(),
): "expired" | "failed" {
  return now < new Date(expiresAt).getTime() - 5_000 ? "failed" : "expired";
}

// Receipt page for a pending/paid record — guests need the token in the URL.
export function buildOrderReturnPath(
  pending: { orderId: string; returnToken: string | null },
): string {
  const params = new URLSearchParams({ orderId: pending.orderId });
  if (pending.returnToken) params.set("token", pending.returnToken);
  return `${ROUTES.ORDERS_RETURN}?${params.toString()}`;
}

// Guest-friendly full order detail (hydrated photos + receipt fields).
// `/me/orders/{id}` is for authed runners; this is for guests landing on
// /orders/return with a share token in the URL.
export async function fetchGuestOrderDetail(
  orderId: string,
  token: string,
): Promise<OrderDetail | null> {
  const qs = `?token=${encodeURIComponent(token)}`;
  return api.get<OrderDetail>(`/orders/${encodeURIComponent(orderId)}${qs}`);
}

// Absolute URL to the bundle-download endpoint. Token-only auth on the BE
// (the endpoint is reachable by top-level <a download> navigation, which
// can't carry the JWT — runners and guests use the same path). FE just
// sets `<a href={url}>` and clicks; the BE streams the ZIP back with a
// Content-Disposition that makes the browser save instead of navigate.
export function buildOrderBundleUrl(orderId: string, token: string): string {
  const qs = `?token=${encodeURIComponent(token)}`;
  return `${API_BASE_URL}/orders/${encodeURIComponent(orderId)}/download-bundle${qs}`;
}
