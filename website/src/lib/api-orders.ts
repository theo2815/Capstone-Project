import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { useOrdersStore, type MockOrder } from "@/store/orders-store";
import type { PaginatedResponse } from "@/types/api";
import type { Order, OrderPhotoDetail } from "@/types/order";

// Orders backend contract (Q-008 idempotency, Q-005 photo URLs RESOLVED 2026-05-09).
//   GET  /api/v1/me/orders?offset=&limit=    → PaginatedResponse<MockOrder>
//   GET  /api/v1/me/orders/{id}              → OrderDetail (MockOrder + photos[] + downloadBundleUrl)
//   POST /api/v1/orders                      { items, paymentMethod, recipientEmail?, idempotencyKey } → Order
//   POST /api/v1/me/orders/{id}/refund       { photoIds, reason, note } → unknown (caller pushes to dispute store)
//
// ADR addendum: list endpoint MUST return `photoIds[]` alongside `photoCount`
// so the runner refund modal opens without a detail round-trip. The spec's
// original "slimmer" OrderSummary omitted photoIds — FE drives the contract here.

export interface OrderDetail extends MockOrder {
  photos: OrderPhotoDetail[];
  downloadBundleUrl?: string;
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

  if (BACKEND_LIVE) {
    const params = new URLSearchParams();
    params.set("offset", String(offset));
    params.set("limit", String(limit));
    const res = await api.get<PaginatedResponse<MockOrder>>(
      `/me/orders?${params.toString()}`,
    );
    return res.items;
  }

  // Mock fallback — read from useOrdersStore. Sort newest-first to match the
  // /orders page expected order (it sorts again, but stable + cheap).
  return [...useOrdersStore.getState().orders].sort((a, b) =>
    (b.paidAt ?? "").localeCompare(a.paidAt ?? ""),
  );
}

export async function fetchOrderDetail(
  orderId: string,
): Promise<OrderDetail | null> {
  if (BACKEND_LIVE) {
    return api.get<OrderDetail>(`/me/orders/${encodeURIComponent(orderId)}`);
  }

  // Mock fallback — synthesize a thin detail from the existing MockOrder.
  // photos[] carries placeholders only; PhotoStrip + lightbox fall back to
  // ID-chip rendering because URL fields are undefined.
  const order = useOrdersStore.getState().orders.find((o) => o.id === orderId);
  if (!order) return null;
  const photos: OrderPhotoDetail[] = order.photoIds.map((id, i) => ({
    id,
    bib: null,
    time: "—",
    tone: i,
  }));
  return { ...order, photos };
}

export async function postOrder(args: CreateOrderArgs): Promise<Order> {
  return api.post<Order>("/orders", args);
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
