"use client";

import { useCartStore } from "@/store/cart-store";
import type { CartItem } from "@/types/order";

export function useCart() {
  const { items, addItem, removeItem, clear, total } = useCartStore();

  return {
    items,
    itemCount: items.length,
    total: total(),
    addItem: (item: CartItem) => addItem(item),
    removeItem: (photoId: string) => removeItem(photoId),
    clear,
    hasItem: (photoId: string) => items.some((i) => i.photoId === photoId),
  };
}
