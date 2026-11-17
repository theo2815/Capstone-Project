import { api } from "@/lib/api";
import type { CartItem } from "@/types/order";

// Cart backend contract (Q-003 RESOLVED 2026-05-09 via merge endpoint).
//   GET    /api/v1/me/cart                 → CartItem[]
//   POST   /api/v1/me/cart/items           { photoId, eventId } → CartItem
//   DELETE /api/v1/me/cart/items/{photoId} → { removed }
//   POST   /api/v1/me/cart/merge           { items } → CartItem[]
//   DELETE /api/v1/me/cart                 → { cleared }
//
// Server returns the canonical row including current price (server-wins on
// price per Q-003).

// includeUnavailable keeps a suspended photographer's rows, marked available: false (AA-I20a).
export async function fetchCart(): Promise<CartItem[]> {
  return api.get<CartItem[]>("/me/cart?includeUnavailable=true");
}

export async function postCartItem(item: CartItem): Promise<CartItem> {
  return api.post<CartItem>("/me/cart/items", {
    photoId: item.photoId,
    eventId: item.eventId,
  });
}

export async function deleteCartItem(photoId: string): Promise<void> {
  await api.delete<{ removed: boolean }>(
    `/me/cart/items/${encodeURIComponent(photoId)}`,
  );
}

export async function mergeCart(localItems: CartItem[]): Promise<CartItem[]> {
  // The merge reply drops unavailable rows, so re-read the cart with them.
  if (localItems.length > 0) await api.post<CartItem[]>("/me/cart/merge", { items: localItems });
  return fetchCart();
}

export async function clearServerCart(): Promise<void> {
  await api.delete<{ cleared: number }>("/me/cart");
}
