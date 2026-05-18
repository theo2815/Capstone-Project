import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { OrderStatus } from "@/types/order";

// Local cache of orders for offline-friendly Race Log derivation.
// Phase E (2026-05-09): /orders page reads via `useOrdersList()` hook which
// pulls from /me/orders. CheckoutModal pushes a fresh MockOrder here on
// payment success so the Race Log derives "bought" rows without waiting
// on a /me/orders refetch.
//
// Initial state is an empty array — no demo seed. The earlier `SEED_ORDERS`
// (QP-DEMO01 + QP-DEMO02) was a development convenience that every fresh
// user inherited, contributing to the "all users share same data" symptom
// fixed 2026-05-18.

export interface MockOrder {
  id: string;            // e.g. "QP-A1B2C3"
  eventId: string;       // links back to event catalog
  photoIds: string[];    // photo IDs that were in the cart at checkout
  total: number;         // peso total
  paymentMethod: string;
  paidAt: string;        // ISO timestamp
  // Phase E backend-hydrated optional fields. Server populates these on the
  // /me/orders list payload; locally-pushed checkout orders leave them
  // undefined and consumers (label helpers, kicker counts) fall back to
  // local catalog joins.
  eventName?: string;
  eventSlug?: string;
  status?: OrderStatus;
}

interface OrdersState {
  orders: MockOrder[];
  setOrders: (orders: MockOrder[]) => void;
  addOrder: (order: MockOrder) => void;
  clear: () => void;
}

export const useOrdersStore = create<OrdersState>()(
  persist(
    (set) => ({
      orders: [],
      setOrders: (orders) => set({ orders }),
      addOrder: (order) =>
        set((state) => ({ orders: [order, ...state.orders] })),
      clear: () => set({ orders: [] }),
    }),
    { name: "quickpitik-orders" },
  ),
);
