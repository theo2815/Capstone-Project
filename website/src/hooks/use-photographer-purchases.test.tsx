import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerPurchasedPhotos, usePhotographerPhotoPurchases } from "./use-photographer-purchases";

let client: QueryClient;
const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
const login = (id = "seller-a") => useAuthStore.getState().setUser({ id, role: "PHOTOGRAPHER", name: id, email: `${id}@example.test`, createdAt: "2026-09-10" });
const envelope = (items: unknown[], offset = 0, total = items.length) => ({ items, offset, limit: 25, total });

beforeEach(() => { useAuthStore.getState().logout(); client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); });
afterEach(() => { client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("opens a pinned purchase beyond page one", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ ...envelope([{ orderId: "order-a", photoId: "photo-a" }], 0, 2), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: { orderId: "order-b", photoId: "photo-a", buyer: "Aira S." } });
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a", "order-b"), { wrapper });
  await waitFor(() => expect(result.current.selectedPurchase?.orderId).toBe("order-b"));
  expect(result.current.items.map((item) => item.orderId)).toEqual(["order-a"]);
  expect(get.mock.calls[0][0]).toContain("orderId=order-b");
  expect(client.getQueryCache().getAll()[0].queryKey).toEqual(["photographer", "photos", "photo-a", "purchases", "order-b", "seller-a", "PHOTOGRAPHER", expect.any(Number)]);
});

it("append failure keeps loaded receipts", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ ...envelope([{ orderId: "order-a", photoId: "photo-a" }], 0, 2), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null }).mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMIT", message: "Wait" }], 429));
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  expect(result.current.items[0]).not.toHaveProperty("id");
  await act(async () => { await result.current.fetchNextPage(); });
  expect(get).toHaveBeenCalledTimes(2);
  expect(result.current.items).toHaveLength(1);
  await waitFor(() => expect(result.current.appendError).toBeTruthy());
});

it("dedupes photo pages by photo ID", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }], 0, 2)).mockResolvedValueOnce(envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }], 1, 2));
  const { result } = renderHook(() => usePhotographerPurchasedPhotos("event-a", true), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  await act(async () => { await result.current.fetchNextPage(); });
  expect(result.current.items).toHaveLength(1);
  expect(get.mock.calls[0][0]).toContain("/events/event-a/purchased-photos");
  expect(get.mock.calls[1][0]).toContain("offset=1");
});
