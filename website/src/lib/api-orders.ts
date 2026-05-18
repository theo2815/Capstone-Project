import { api } from "@/lib/api";
import type { MockOrder } from "@/store/orders-store";
import type { PaginatedResponse } from "@/types/api";
import type { Order, OrderPhotoDetail, OrderStatus } from "@/types/order";

// Orders backend contract (Q-008 idempotency, Q-005 photo URLs RESOLVED 2026-05-09).
//   GET  /api/v1/me/orders?offset=&limit=    → PaginatedResponse<MockOrder>
//   GET  /api/v1/me/orders/{id}              → OrderDetail (MockOrder + photos[] + downloadBundleUrl)
//   POST /api/v1/orders                      { items, paymentMethod, recipientEmail? } + Idempotency-Key header → Order
//   POST /api/v1/me/orders/{id}/refund       { photoIds, reason, note } → unknown
//
// Idempotency-Key arrives as an HTTP header per RFC 9110 §9.2.2.

export interface OrderDetail extends MockOrder {
  photos: OrderPhotoDetail[];
  downloadBundleUrl?: string;
  // Populated for the /orders/return success state. May be empty on legacy
  // payloads or when the BE chose to omit it.
  recipientEmail?: string;
}

export interface CreateOrderArgs {
  items: { photoId: string; eventId: string }[];
  paymentMethod: "gcash" | "maya" | "card";
  recipientEmail?: string;
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

// Guest-friendly status poll. The /orders/return page calls this to check
// whether PayMongo's webhook has fired yet. Authed runners hit
// `fetchOrderDetail` instead — they get full hydration via JWT.
export interface OrderStatusPayload {
  id: string;
  status: OrderStatus;
  paidAt: string | null;
}

export async function fetchOrderStatus(
  orderId: string,
  token: string,
): Promise<OrderStatusPayload> {
  const qs = `?token=${encodeURIComponent(token)}`;
  return api.get<OrderStatusPayload>(
    `/orders/${encodeURIComponent(orderId)}/status${qs}`,
  );
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
