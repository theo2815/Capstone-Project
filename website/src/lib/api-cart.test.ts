import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { fetchCart, mergeCart } from "./api-cart";

afterEach(() => vi.restoreAllMocks());

const local = { photoId: "p1", eventId: "e1", thumbnailUrl: "", price: 125 };

it("fetchCart asks for includeUnavailable=true", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue([]);
  await fetchCart();
  expect(get).toHaveBeenCalledWith("/me/cart?includeUnavailable=true");
});

it("mergeCart re-reads the cart with unavailable rows after merging", async () => {
  const unavailable = { ...local, photoId: "p2", available: false };
  const post = vi.spyOn(api, "post").mockResolvedValue([local]);
  vi.spyOn(api, "get").mockResolvedValue([local, unavailable]);

  expect(await mergeCart([local])).toEqual([local, unavailable]);
  expect(post).toHaveBeenCalledWith("/me/cart/merge", { items: [local] });
});
