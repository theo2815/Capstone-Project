"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchOrders,
  fetchOrderDetail,
  type OrderDetail,
} from "@/lib/api-orders";
import type { OrderSummary } from "@/types/order";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";

const EMPTY_ORDERS: OrderSummary[] = [];

// Full order list (capped at the fetcher default). Both /orders slabs share it:
// SpendSlab needs every order to compute lifetime totals, so ReceiptsSlab
// client-slices the same in-memory set rather than issuing a second fetch. True
// receipts pagination is blocked on a BE order-summary endpoint to decouple the
// spend aggregate — filed under "requires backend changes".
export function useOrdersList(): {
  orders: OrderSummary[];
  isLoading: boolean;
  error: unknown;
} {
  const session = usePrivateQuerySession();
  const query = useQuery<OrderSummary[]>({
    queryKey: ["me", "orders", "list", ...session.key],
    queryFn: ({ signal }) => fetchOrders({}, signal),
    enabled: session.enabled,
    staleTime: 30_000,
  });

  return {
    orders: query.data ?? EMPTY_ORDERS,
    isLoading: query.isPending && session.enabled,
    error: query.error,
  };
}

export function useOrderDetail(orderId: string | null): {
  detail: OrderDetail | null;
  isLoading: boolean;
  error: unknown;
} {
  const session = usePrivateQuerySession();
  const query = useQuery<OrderDetail | null>({
    queryKey: ["me", "orders", "detail", orderId, ...session.key],
    queryFn: ({ signal }) => (orderId ? fetchOrderDetail(orderId, signal) : Promise.resolve(null)),
    enabled: session.enabled && !!orderId,
    staleTime: 60_000,
  });

  return {
    detail: query.data ?? null,
    isLoading: query.isPending && session.enabled && !!orderId,
    error: query.error,
  };
}
