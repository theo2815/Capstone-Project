"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  fetchOrders,
  fetchOrderDetail,
  type OrderDetail,
} from "@/lib/api-orders";
import { useOrdersStore, type MockOrder } from "@/store/orders-store";
import type { OrderPhotoDetail } from "@/types/order";

// React Query for live mode, Zustand subscription for mock mode. Two render
// paths so mock-mode mutations (CheckoutModal addOrder) propagate without a
// manual cache invalidation.

export function useOrdersList(): {
  orders: MockOrder[];
  isLoading: boolean;
  error: unknown;
} {
  const storeOrders = useOrdersStore((s) => s.orders);
  const sortedStoreOrders = useMemo(
    () =>
      [...storeOrders].sort((a, b) =>
        (b.paidAt ?? "").localeCompare(a.paidAt ?? ""),
      ),
    [storeOrders],
  );

  const query = useQuery<MockOrder[]>({
    queryKey: ["me", "orders"],
    queryFn: () => fetchOrders(),
    enabled: BACKEND_LIVE,
    staleTime: 30_000,
  });

  if (BACKEND_LIVE) {
    return {
      orders: query.data ?? [],
      isLoading: query.isPending,
      error: query.error,
    };
  }
  return { orders: sortedStoreOrders, isLoading: false, error: null };
}

export function useOrderDetail(orderId: string | null): {
  detail: OrderDetail | null;
  isLoading: boolean;
  error: unknown;
} {
  const storeOrders = useOrdersStore((s) => s.orders);

  const query = useQuery<OrderDetail | null>({
    queryKey: ["me", "orders", orderId],
    queryFn: () => (orderId ? fetchOrderDetail(orderId) : Promise.resolve(null)),
    enabled: BACKEND_LIVE && orderId !== null,
    staleTime: 60_000,
  });

  // Mock-mode synthetic detail (must run unconditionally to satisfy hook order).
  const mockOrder = orderId
    ? storeOrders.find((o) => o.id === orderId)
    : undefined;
  const mockDetail: OrderDetail | null = useMemo(() => {
    if (!mockOrder) return null;
    const photos: OrderPhotoDetail[] = mockOrder.photoIds.map((id, i) => ({
      id,
      bib: null,
      time: "—",
      tone: i,
    }));
    return { ...mockOrder, photos };
  }, [mockOrder]);

  if (BACKEND_LIVE) {
    return {
      detail: query.data ?? null,
      isLoading: query.isPending,
      error: query.error,
    };
  }
  return { detail: mockDetail, isLoading: false, error: null };
}
