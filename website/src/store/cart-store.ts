import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { CartItem } from "@/types/order";

interface CartState {
  items: CartItem[];
  addItem: (item: CartItem) => void;
  removeItem: (photoId: string) => void;
  clear: () => void;
  total: () => number;
}

export const useCartStore = create<CartState>()(
  persist(
    (set, get) => ({
      items: [],
      addItem: (item) =>
        set((state) => {
          if (state.items.some((i) => i.photoId === item.photoId))
            return state;
          return { items: [...state.items, item] };
        }),
      removeItem: (photoId) =>
        set((state) => ({
          items: state.items.filter((i) => i.photoId !== photoId),
        })),
      clear: () => set({ items: [] }),
      total: () => get().items.reduce((sum, item) => sum + item.price, 0),
    }),
    { name: "eventai-cart" },
  ),
);
