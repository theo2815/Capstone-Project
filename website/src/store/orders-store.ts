import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { OrderStatus } from "@/types/order";

// Local cache of orders for mock-mode + offline-friendly Race Log derivation.
// Phase E (2026-05-09): /orders page reads via `useOrdersList()` hook which
// pulls from /me/orders in live mode and falls back here in mock mode. The
// CheckoutModal mock branch still pushes a MockOrder here on payment success.

export interface MockOrder {
  id: string;            // e.g. "QP-A1B2C3"
  eventId: string;       // links back to event catalog
  photoIds: string[];    // photo IDs that were in the cart at checkout
  total: number;         // peso total
  paymentMethod: string;
  paidAt: string;        // ISO timestamp
  // Phase E backend-hydrated optional fields. Server populates these on the
  // /me/orders list payload; mock seed leaves them undefined and consumers
  // (label helpers, kicker counts) fall back to local catalog joins.
  eventName?: string;
  eventSlug?: string;
  status?: OrderStatus;
}

// Demo seed so a fresh user sees Race Log "kept" counts immediately.
// - Order on event "1" (Cebu Marathon)        → pairs with saved → save+bought row
// - Order on event "3" (SRP Half-Marathon)    → never saved → bought-only row (retroactive entry)
const SEED_ORDERS: ReadonlyArray<MockOrder> = [
  {
    id: "QP-DEMO01",
    eventId: "1",
    photoIds: ["mock-cm-1", "mock-cm-2", "mock-cm-3"],
    total: 240,
    paymentMethod: "gcash",
    paidAt: "2026-04-29T10:23:00.000Z",
  },
  {
    id: "QP-DEMO02",
    eventId: "3",
    photoIds: ["mock-srp-1", "mock-srp-2"],
    total: 140,
    paymentMethod: "gcash",
    paidAt: "2026-04-13T18:11:00.000Z",
  },
];

interface OrdersState {
  orders: MockOrder[];
  setOrders: (orders: MockOrder[]) => void;
  addOrder: (order: MockOrder) => void;
  clear: () => void;
}

export const useOrdersStore = create<OrdersState>()(
  persist(
    (set) => ({
      orders: [...SEED_ORDERS],
      setOrders: (orders) => set({ orders }),
      addOrder: (order) =>
        set((state) => ({ orders: [order, ...state.orders] })),
      clear: () => set({ orders: [] }),
    }),
    { name: "quickpitik-orders" },
  ),
);
