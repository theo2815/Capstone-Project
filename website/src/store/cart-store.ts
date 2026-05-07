import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CartItem } from "@/types/order";

// Hard ceiling on cart size. Photos are deduped by `photoId` so per-photo
// quantity is always 1 — this cap protects against BuyAllBar dumping a
// massive race into the cart in one click. Reasonable headroom for keepers
// across multiple events without becoming a checkout-time foot-gun.
export const MAX_CART_ITEMS = 100;

interface CartState {
  items: CartItem[];
  addItem: (item: CartItem) => boolean;
  removeItem: (photoId: string) => void;
  clear: () => void;
  total: () => number;
  isFull: () => boolean;
  remainingCapacity: () => number;
}

export const useCartStore = create<CartState>()(
  persist(
    (set, get) => ({
      items: [],
      addItem: (item) => {
        let added = false;
        set((state) => {
          if (state.items.some((i) => i.photoId === item.photoId)) return state;
          if (state.items.length >= MAX_CART_ITEMS) return state;
          added = true;
          return { items: [...state.items, item] };
        });
        return added;
      },
      removeItem: (photoId) =>
        set((state) => ({
          items: state.items.filter((i) => i.photoId !== photoId),
        })),
      clear: () => set({ items: [] }),
      total: () => get().items.reduce((sum, item) => sum + item.price, 0),
      isFull: () => get().items.length >= MAX_CART_ITEMS,
      remainingCapacity: () =>
        Math.max(0, MAX_CART_ITEMS - get().items.length),
    }),
    { name: "quickpitik-cart" },
  ),
);
