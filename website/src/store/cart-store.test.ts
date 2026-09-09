import { describe, expect, it, beforeEach } from "vitest";
import { useCartStore } from "./cart-store";
import type { CartItem } from "@/types/order";

function makeItem(photoId: string, price: number): CartItem {
  return {
    photoId,
    eventId: "event-01",
    thumbnailUrl: "https://example.com/thumb.jpg",
    price,
  };
}

describe("useCartStore pricing precision", () => {
  beforeEach(() => {
    useCartStore.setState({ items: [], syncEnabled: false });
  });

  it("calculates exact total without floating point accumulation issues", () => {
    // 150.50 + 99.99 in naive JS floats is 250.49000000000004
    useCartStore.getState().addItem(makeItem("p1", 150.50));
    useCartStore.getState().addItem(makeItem("p2", 99.99));

    expect(useCartStore.getState().total()).toBe(250.49);
  });

  it("handles multiple micro-decimal additions precisely", () => {
    useCartStore.getState().addItem(makeItem("p1", 19.99));
    useCartStore.getState().addItem(makeItem("p2", 29.99));
    useCartStore.getState().addItem(makeItem("p3", 49.99));

    // 19.99 + 29.99 + 49.99 = 99.97
    expect(useCartStore.getState().total()).toBe(99.97);
  });

  it("returns 0 for empty cart", () => {
    expect(useCartStore.getState().total()).toBe(0);
  });
});
