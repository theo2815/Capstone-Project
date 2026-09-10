"use client";

import { useMemo } from "react";
import { useOrdersList } from "@/hooks/use-orders";

// Early duplicate-purchase feedback shares the canonical order list. Guests
// have no order-history endpoint; the backend still enforces guest-email and
// pending-order overlap checks at checkout.
export function useOwnedPhotoIds(): Set<string> {
  const { orders } = useOrdersList();
  return useMemo(() => new Set(
    orders
      .filter((order) => order.status === "PAID" || order.status === "FULFILLED")
      .flatMap((order) => order.photoIds),
  ), [orders]);
}
