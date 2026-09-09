"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchOrders } from "@/lib/api-orders";
import { useAuthStore } from "@/store/auth-store";
import { useOrdersStore, type MockOrder } from "@/store/orders-store";

// Photo IDs the current viewer already owns — i.e. a PAID/FULFILLED order
// covers the photo. This reproduces the backend's already-purchased check
// (OrderService.reserveCheckout) client-side so the cart / checkout can flag a
// duplicate BEFORE the runner tries to pay, instead of on the 409.
//
// Signed-in: authoritative from GET /me/orders (shares the ["me","orders"]
// query the checkout invalidates on success). Guests have no list endpoint, so
// we fall back to the locally-persisted orders store for a same-browser buy.
// Both are strictly narrower than the server gate (which also blocks PENDING
// overlap + guest-email), so the backend 409 stays the net.
const OWNED_STATUSES = new Set<string>(["PAID", "FULFILLED"]);

function collectOwned(orders: MockOrder[], into: Set<string>): void {
  for (const order of orders) {
    // A locally-pushed order leaves status undefined — treat it as owned
    // (it only lands there on a completed purchase); a hydrated order must be
    // PAID/FULFILLED to count.
    if (order.status && !OWNED_STATUSES.has(order.status)) continue;
    for (const photoId of order.photoIds ?? []) into.add(photoId);
  }
}

export function useOwnedPhotoIds(): Set<string> {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const { data } = useQuery<MockOrder[]>({
    queryKey: ["me", "orders"],
    // Coerce to an array so an empty/odd response can't make React Query warn
    // about undefined data or make collectOwned iterate a non-array.
    queryFn: async () => {
      const orders = await fetchOrders();
      return Array.isArray(orders) ? orders : [];
    },
    enabled: isAuthenticated, // guests would only 401 here
    staleTime: 30_000,
  });
  const localOrders = useOrdersStore((s) => s.orders);

  return useMemo(() => {
    const owned = new Set<string>();
    collectOwned(data ?? [], owned);
    collectOwned(localOrders, owned);
    return owned;
  }, [data, localOrders]);
}
