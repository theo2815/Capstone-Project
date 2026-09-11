"use client";

import { useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  postOrder,
  fetchPendingStatus,
  cancelPendingPayment,
  type OrderStatusPayload,
  type CreateOrderArgs,
  type OrderStatusOptions,
} from "@/lib/api-orders";
import type { Order } from "@/types/order";
import { useAuthStore } from "@/store/auth-store";

export interface CheckoutPendingRecord {
  orderId: string;
  returnToken: string | null;
}

export const checkoutKeys = {
  orders: ["me", "orders"] as const,
};

function sessionSnapshot() {
  const session = useAuthStore.getState();
  return {
    isAuthenticated: session.isAuthenticated,
    userId: session.user?.id ?? null,
    userRole: session.user?.role ?? null,
    sessionVersion: session.sessionVersion,
  };
}

function isCurrentSession(snapshot: ReturnType<typeof sessionSnapshot>) {
  const current = useAuthStore.getState();
  return (
    current.isAuthenticated === snapshot.isAuthenticated &&
    current.sessionVersion === snapshot.sessionVersion &&
    (current.user?.id ?? null) === snapshot.userId &&
    (current.user?.role ?? null) === snapshot.userRole
  );
}

export function useCheckoutActions() {
  const queryClient = useQueryClient();

  return useMemo(() => {
    const createOrder = async (args: CreateOrderArgs): Promise<Order | null> => {
      const snapshot = sessionSnapshot();
      const order = await postOrder(args);
      if (!isCurrentSession(snapshot)) return null;
      await queryClient.invalidateQueries({ queryKey: checkoutKeys.orders });
      return order;
    };

    const checkStatus = async (
      pending: CheckoutPendingRecord,
      opts: OrderStatusOptions = {},
    ): Promise<OrderStatusPayload | null> => {
      const snapshot = sessionSnapshot();
      const status = await fetchPendingStatus(pending, opts);
      if (!isCurrentSession(snapshot)) return null;
      return status;
    };

    const cancelPayment = async (
      pending: CheckoutPendingRecord,
    ): Promise<OrderStatusPayload | null> => {
      const snapshot = sessionSnapshot();
      const status = await cancelPendingPayment(pending);
      if (!isCurrentSession(snapshot)) return null;
      return status;
    };

    return { createOrder, checkStatus, cancelPayment };
  }, [queryClient]);
}
