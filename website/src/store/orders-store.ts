import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { RunnerDispute } from "@/lib/api-orders";
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
  // 2026-05-19 PM — date + state added so /profile race log can render
  // purchased-only rows (events bought but not saved) without a second
  // round-trip per event. Optional because locally-pushed checkout orders
  // and legacy payloads omit them; race log skips the date/state label
  // when null and falls back to whatever the catalog can provide.
  eventDate?: string;
  eventState?: "upcoming" | "live" | "open" | "past";
  status?: OrderStatus;
  // Photographer coupon (V45): the code entered at checkout and the sum of
  // per-item discounts. `total` is already what was charged.
  couponCode?: string | null;
  discountTotal?: number;
  // Refund disputes filed against this order. Source of truth for the
  // /orders status chip, photo locks, timeline, and cancel button. Embedded
  // on every BE payload from /me/orders since 2026-05-19; locally-pushed
  // checkout orders default to an empty array.
  disputes?: RunnerDispute[];
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
