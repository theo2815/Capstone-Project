import { captureAuthSession } from "@/lib/auth-session";
import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CartItem } from "@/types/order";
import {
  postCartItem,
  deleteCartItem,
  clearServerCart,
  fetchCart,
} from "@/lib/api-cart";
import { fetchLivePrices } from "@/lib/api-coupons";

// Hard ceiling on cart size. Photos are deduped by `photoId` so per-photo
// quantity is always 1 — this cap protects against BuyAllBar dumping a
// massive race into the cart in one click. Reasonable headroom for keepers
// across multiple events without becoming a checkout-time foot-gun.
export const MAX_CART_ITEMS = 100;

/** The rows that can be paid for: a suspended photographer's rows stay in the cart but out of the total and checkout. */
export function payableItems(items: CartItem[]): CartItem[] {
  return items.filter((i) => i.available !== false);
}

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
  /** Re-reads what checkout charges: the server cart when signed in, live per-photo prices for a guest. */
  refreshPrices: () => Promise<void>;
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
          // A guest row keeps its add-time price; a signed-in row takes the server's.
          return { items: [...state.items, { ...item, priceAtAdd: item.priceAtAdd ?? item.price }] };
        });
        if (added && get().syncEnabled) {
          const isCurrent = captureAuthSession();
          postCartItem(item)
            .then((canonical) => {
              if (!isCurrent()) return;
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
            if (!isCurrent()) return;
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
          const isCurrent = captureAuthSession();
          const reverted = removed;
          deleteCartItem(photoId).catch(() => {
            if (!isCurrent()) return;
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
      refreshPrices: async () => {
        const isCurrent = captureAuthSession();
        if (get().syncEnabled) {
          const rows = await fetchCart();
          if (isCurrent()) set({ items: rows });
          return;
        }
        const ids = get().items.map((i) => i.photoId);
        if (ids.length === 0) return;
        const live = await fetchLivePrices(ids);
        if (!isCurrent() || live.size === 0) return;
        // Re-read at write time: an add or remove may have landed meanwhile. Only `price` changes.
        set((state) => ({
          items: state.items.map((i) => {
            const price = live.get(i.photoId);
            return price === undefined || price === i.price ? i : { ...i, price };
          }),
        }));
      },
      total: () => {
        const cents = payableItems(get().items).reduce(
          (sum, item) => sum + Math.round(Number(item.price || 0) * 100),
          0,
        );
        return cents / 100;
      },
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
