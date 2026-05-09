import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CartItem } from "@/types/order";
import {
  postCartItem,
  deleteCartItem,
  clearServerCart,
} from "@/lib/api-cart";

// Hard ceiling on cart size. Photos are deduped by `photoId` so per-photo
// quantity is always 1 — this cap protects against BuyAllBar dumping a
// massive race into the cart in one click. Reasonable headroom for keepers
// across multiple events without becoming a checkout-time foot-gun.
export const MAX_CART_ITEMS = 100;

interface CartState {
  items: CartItem[];
  // When true, mutators mirror to /me/cart. Set by `<AuthHydrator>` after the
  // post-login merge resolves; cleared on logout. While false, the store is
  // local-only (guest mode + offline buffer).
  syncEnabled: boolean;
  setSyncEnabled: (enabled: boolean) => void;
  setItems: (items: CartItem[]) => void;
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
      syncEnabled: false,
      setSyncEnabled: (syncEnabled) => set({ syncEnabled }),
      setItems: (items) => set({ items }),
      addItem: (item) => {
        let added = false;
        set((state) => {
          if (state.items.some((i) => i.photoId === item.photoId)) return state;
          if (state.items.length >= MAX_CART_ITEMS) return state;
          added = true;
          return { items: [...state.items, item] };
        });
        if (added && get().syncEnabled) {
          postCartItem(item)
            .then((canonical) => {
              // Server-wins on price (Q-003). Patch the local row if the
              // returned price differs — keeps checkout total honest.
              if (canonical && canonical.price !== item.price) {
                set((state) => ({
                  items: state.items.map((i) =>
                    i.photoId === item.photoId ? { ...i, ...canonical } : i,
                  ),
                }));
              }
            })
            .catch(() => {
              // Revert optimistic add on failure.
              set((state) => ({
                items: state.items.filter((i) => i.photoId !== item.photoId),
              }));
            });
        }
        return added;
      },
      removeItem: (photoId) => {
        let removed: CartItem | undefined;
        set((state) => {
          const idx = state.items.findIndex((i) => i.photoId === photoId);
          if (idx === -1) return state;
          removed = state.items[idx];
          return { items: state.items.filter((i) => i.photoId !== photoId) };
        });
        if (removed && get().syncEnabled) {
          const reverted = removed;
          deleteCartItem(photoId).catch(() => {
            // Revert optimistic delete on failure.
            set((state) => ({ items: [...state.items, reverted] }));
          });
        }
      },
      clear: () => {
        set({ items: [] });
        if (get().syncEnabled) {
          clearServerCart().catch(() => {
            // Server-side clear is opportunistic — if it fails the local cart
            // stays empty (matches user intent) and the next login merge will
            // reconcile.
          });
        }
      },
      total: () => get().items.reduce((sum, item) => sum + item.price, 0),
      isFull: () => get().items.length >= MAX_CART_ITEMS,
      remainingCapacity: () =>
        Math.max(0, MAX_CART_ITEMS - get().items.length),
    }),
    {
      name: "quickpitik-cart",
      // syncEnabled is a runtime flag (set by AuthHydrator), never persist it.
      partialize: (state) => ({ items: state.items }),
    },
  ),
);
