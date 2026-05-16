"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchOrders,
  fetchOrderDetail,
  type OrderDetail,
} from "@/lib/api-orders";
import type { MockOrder } from "@/store/orders-store";

export function useOrdersList(): {
  orders: MockOrder[];
  isLoading: boolean;
  error: unknown;
} {
  const query = useQuery<MockOrder[]>({
    queryKey: ["me", "orders"],
    queryFn: () => fetchOrders(),
    staleTime: 30_000,
  });

  return {
    orders: query.data ?? [],
    isLoading: query.isPending,
    error: query.error,
  };
}

export function useOrderDetail(orderId: string | null): {
  detail: OrderDetail | null;
  isLoading: boolean;
  error: unknown;
} {
  const query = useQuery<OrderDetail | null>({
    queryKey: ["me", "orders", orderId],
    queryFn: () => (orderId ? fetchOrderDetail(orderId) : Promise.resolve(null)),
    enabled: orderId !== null,
    staleTime: 60_000,
  });

  return {
    detail: query.data ?? null,
    isLoading: query.isPending,
    error: query.error,
  };
}
