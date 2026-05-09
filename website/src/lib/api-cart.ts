import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import type { CartItem } from "@/types/order";

// Cart backend contract (Q-003 RESOLVED 2026-05-09 via merge endpoint).
//   GET    /api/v1/me/cart                 → CartItem[]
//   POST   /api/v1/me/cart/items           { photoId, eventId } → CartItem
//   DELETE /api/v1/me/cart/items/{photoId} → { removed }
//   POST   /api/v1/me/cart/merge           { items } → CartItem[]
//   DELETE /api/v1/me/cart                 → { cleared }
//
// Server returns the canonical row including current price (server-wins on
// price per Q-003). Mock-mode echoes input so optimistic local updates
// still work.

export async function fetchCart(): Promise<CartItem[]> {
  if (BACKEND_LIVE) return api.get<CartItem[]>("/me/cart");
  return [];
}

export async function postCartItem(item: CartItem): Promise<CartItem> {
  if (!BACKEND_LIVE) return item;
  return api.post<CartItem>("/me/cart/items", {
    photoId: item.photoId,
    eventId: item.eventId,
  });
}

export async function deleteCartItem(photoId: string): Promise<void> {
  if (!BACKEND_LIVE) return;
  await api.delete<{ removed: boolean }>(
    `/me/cart/items/${encodeURIComponent(photoId)}`,
  );
}

export async function mergeCart(localItems: CartItem[]): Promise<CartItem[]> {
  if (BACKEND_LIVE) {
    if (localItems.length === 0) return api.get<CartItem[]>("/me/cart");
    return api.post<CartItem[]>("/me/cart/merge", { items: localItems });
  }
  return localItems;
}

export async function clearServerCart(): Promise<void> {
  if (!BACKEND_LIVE) return;
  await api.delete<{ cleared: number }>("/me/cart");
}
