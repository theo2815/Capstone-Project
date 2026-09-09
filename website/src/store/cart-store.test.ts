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
    // The previous direct reduction returns 0.30000000000000004.
    useCartStore.getState().addItem(makeItem("p1", 0.10));
    useCartStore.getState().addItem(makeItem("p2", 0.20));

    expect(useCartStore.getState().total()).toBe(0.30);
  });

  it("handles multiple micro-decimal additions precisely", () => {
    for (let i = 0; i < 10; i++) {
      useCartStore.getState().addItem(makeItem(`p${i}`, 0.10));
    }
    expect(useCartStore.getState().total()).toBe(1);
  });

  it("returns 0 for empty cart", () => {
    expect(useCartStore.getState().total()).toBe(0);
  });
});
