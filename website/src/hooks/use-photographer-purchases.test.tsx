import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useLayoutEffect, type ReactNode } from "react";
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
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ ...envelope([{ orderId: "order-a", photoId: "photo-a" }], 0, 100), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null }).mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMIT", message: "Wait" }], 429));
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  expect(result.current.items[0]).not.toHaveProperty("id");
  await act(async () => { await result.current.fetchNextPage(); });
  expect(get).toHaveBeenCalledTimes(2);
  expect(result.current.items).toHaveLength(1);
  await waitFor(() => expect(result.current.appendError).toBeTruthy());
});

it.each([404, 429])("seller queries keep HTTP %s separate from offline pause without automatic retry", async (status) => {
  login();
  const get = vi.spyOn(api, "get").mockRejectedValue(new ApiError([{ code: "UNAVAILABLE", message: "Unavailable" }], status));
  const { result } = renderHook(() => ({
    gallery: usePhotographerPurchasedPhotos("event-a", true),
    history: usePhotographerPhotoPurchases("photo-a", "order-a"),
  }), { wrapper });
  await waitFor(() => expect(result.current.gallery.error).toBeInstanceOf(ApiError));
  await waitFor(() => expect(result.current.history.error).toBeInstanceOf(ApiError));
  expect(result.current.gallery.isPaused).toBe(false);
  expect(result.current.history.isPaused).toBe(false);
  expect(get).toHaveBeenCalledTimes(2);
});

it("dedupes photo pages by photo ID", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }], 0, 100)).mockResolvedValueOnce(envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }], 25, 100));
  const { result } = renderHook(() => usePhotographerPurchasedPhotos("event-a", true), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  await act(async () => { await result.current.fetchNextPage(); });
  expect(result.current.items).toHaveLength(1);
  expect(get.mock.calls[0][0]).toContain("/events/event-a/purchased-photos");
  expect(get.mock.calls[1][0]).toContain("offset=25");
});

it("manual controls cannot fetch disabled seller lists", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValue(envelope([]));
  const { result } = renderHook(() => ({
    photos: usePhotographerPurchasedPhotos("event-a", false),
    receipts: usePhotographerPhotoPurchases("photo-a", "order-a", false),
  }), { wrapper });
  await act(async () => {
    await result.current.photos.refetch();
    await result.current.photos.fetchNextPage();
    await result.current.receipts.refetch();
    await result.current.receipts.fetchNextPage();
  });
  expect(get).not.toHaveBeenCalled();
});

it("stale manual controls cannot read seller data after a role switch", async () => {
  login();
  const get = vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.includes("purchased-photos")
    ? envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }], 0, 2)
    : { ...envelope([{ orderId: "order-a", photoId: "photo-a" }], 0, 2), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null }));
  const { result } = renderHook(() => ({
    photos: usePhotographerPurchasedPhotos("event-a", true),
    receipts: usePhotographerPhotoPurchases("photo-a", "order-a"),
  }), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  const oldControls = result.current;
  act(() => useAuthStore.getState().setUser({ id: "runner-a", role: "RUNNER", name: "Runner", email: "runner@example.test", createdAt: "2026-09-10" }));
  await act(async () => {
    await oldControls.photos.refetch();
    await oldControls.photos.fetchNextPage();
    await oldControls.receipts.refetch();
    await oldControls.receipts.fetchNextPage();
  });
  expect(get).toHaveBeenCalledTimes(2);
});

it("layout-effect controls cannot fetch after both seller lists are disabled", async () => {
  login();
  const get = vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.includes("purchased-photos")
    ? envelope([{ photo: { id: "photo-a" }, purchaseCount: 1 }])
    : { ...envelope([{ orderId: "order-a", photoId: "photo-a" }]), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null }));
  const { rerender } = renderHook(({ enabled }) => {
    const photos = usePhotographerPurchasedPhotos("event-a", enabled);
    const receipts = usePhotographerPhotoPurchases("photo-a", "order-a", enabled);
    useLayoutEffect(() => {
      if (!enabled) {
        void photos.refetch();
        void photos.fetchNextPage();
        void receipts.refetch();
        void receipts.fetchNextPage();
      }
    }, [enabled, photos, receipts]);
    return { photos, receipts };
  }, { initialProps: { enabled: true }, wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  await act(async () => { rerender({ enabled: false }); await Promise.resolve(); });
  expect(get).toHaveBeenCalledTimes(2);
});

it("stops purchased-photo pagination on an empty page with a stale total", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValue(envelope([], 0, 100));
  const { result } = renderHook(() => usePhotographerPurchasedPhotos("event-a", true), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(result.current.isLoading).toBe(false));
  expect(result.current.hasNextPage).toBe(false);
});

it("uses the backend page limit for the next purchased-photo offset", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ ...envelope([{ photo: { id: "photo-a" } }], 0, 100), limit: 60 }).mockResolvedValueOnce(envelope([], 60, 100));
  const { result } = renderHook(() => usePhotographerPurchasedPhotos("event-a", true), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  await act(async () => { await result.current.fetchNextPage(); });
  expect(get.mock.calls[1][0]).toContain("offset=60");
});

it("stops receipt pagination on an empty page with a stale total", async () => {
  login();
  const get = vi.spyOn(api, "get").mockResolvedValue({ ...envelope([], 0, 100), photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null });
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(result.current.isLoading).toBe(false));
  expect(result.current.hasNextPage).toBe(false);
});

it("uses the backend page limit for the next receipt offset", async () => {
  login();
  const page = { photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null };
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ ...envelope([{ orderId: "order-a", photoId: "photo-a" }], 0, 100), ...page }).mockResolvedValueOnce({ ...envelope([], 25, 100), ...page });
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  await act(async () => { await result.current.fetchNextPage(); });
  expect(get.mock.calls[1][0]).toContain("offset=25");
});

it("refreshes only page zero and preserves loaded receipt pages and selection", async () => {
  login();
  const page = (items: Array<{ orderId: string; photoId: string }>, offset: number, selectedPurchase: { orderId: string; photoId: string } | null, url: string) => ({
    items, offset, limit: 1, total: 2,
    photo: { id: "photo-a", thumbnailUrl: url },
    event: { id: "event-a", name: offset ? "Old event" : "Cebu Run" },
    selectedPurchase,
  });
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(page([{ orderId: "order-a", photoId: "photo-a" }], 0, { orderId: "order-a", photoId: "photo-a" }, "expired"))
    .mockResolvedValueOnce(page([{ orderId: "order-b", photoId: "photo-a" }], 1, null, "expired"))
    .mockResolvedValueOnce(page([{ orderId: "replacement", photoId: "photo-a" }], 0, { orderId: "order-b", photoId: "photo-a" }, "fresh"));
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a", "order-a"), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  await act(async () => { await result.current.fetchNextPage(); });
  await waitFor(() => expect(result.current.items.map((item) => item.orderId)).toEqual(["order-a", "order-b"]));

  await act(async () => { await result.current.refreshPhoto("order-b"); });

  expect(get).toHaveBeenCalledTimes(3);
  expect(get.mock.calls[2][0]).toContain("offset=0");
  expect(get.mock.calls[2][0]).toContain("limit=1");
  expect(get.mock.calls[2][0]).toContain("orderId=order-b");
  expect(result.current.photo?.thumbnailUrl).toBe("fresh");
  expect(result.current.event?.name).toBe("Cebu Run");
  expect(result.current.items.map((item) => item.orderId)).toEqual(["order-a", "order-b"]);
  expect(result.current.total).toBe(2);
  expect(result.current.selectedPurchase?.orderId).toBe("order-a");
});

it.each(["close", "account switch"])("drops a delayed photo refresh after %s", async (change) => {
  login();
  const initial = (url: string) => ({
    items: [{ orderId: "order-a", photoId: "photo-a" }], offset: 0, limit: 25, total: 1,
    photo: { id: "photo-a", thumbnailUrl: url }, event: { id: "event-a" },
    selectedPurchase: { orderId: "order-a", photoId: "photo-a" },
  });
  let resolveRefresh!: (value: ReturnType<typeof initial>) => void;
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(initial("expired"))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }))
    .mockResolvedValue(initial("other-account"));
  const { result, rerender } = renderHook(({ enabled }) => usePhotographerPhotoPurchases("photo-a", "order-a", enabled), {
    initialProps: { enabled: true }, wrapper,
  });
  await waitFor(() => expect(result.current.photo?.thumbnailUrl).toBe("expired"));
  const oldKey = client.getQueryCache().getAll()[0].queryKey;
  let refresh!: Promise<void>;
  act(() => { refresh = result.current.refreshPhoto(); });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  const signal = get.mock.calls[1][1]?.signal;

  if (change === "close") rerender({ enabled: false });
  else act(() => login("seller-b"));
  expect(signal?.aborted).toBe(true);
  await act(async () => { resolveRefresh(initial("fresh")); await refresh; });

  expect((client.getQueryData<{ pages: ReturnType<typeof initial>[] }>(oldKey))?.pages[0].photo.thumbnailUrl).toBe("expired");
});

it("single-flights a failed photo refresh and allows a bounded retry", async () => {
  login();
  const page = (url: string) => ({
    items: [{ orderId: "order-a", photoId: "photo-a" }], offset: 0, limit: 25, total: 1,
    photo: { id: "photo-a", thumbnailUrl: url }, event: { id: "event-a" }, selectedPurchase: null,
  });
  let rejectRefresh!: (error: Error) => void;
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(page("expired"))
    .mockImplementationOnce(() => new Promise((_, reject) => { rejectRefresh = reject; }))
    .mockResolvedValueOnce(page("fresh"));
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(result.current.photo?.thumbnailUrl).toBe("expired"));

  let first!: Promise<void>;
  let duplicate!: Promise<void>;
  act(() => {
    first = result.current.refreshPhoto("order-a");
    duplicate = result.current.refreshPhoto("order-b");
  });
  expect(duplicate).toBe(first);
  expect(get).toHaveBeenCalledTimes(2);
  const rejected = expect(first).rejects.toThrow("expired signer");
  await act(async () => rejectRefresh(new Error("expired signer")));
  await rejected;
  await waitFor(() => expect(result.current.photoRefreshError).toBeInstanceOf(Error));
  expect(result.current.photo?.thumbnailUrl).toBe("expired");

  await act(async () => { await result.current.refreshPhoto("order-b"); });
  expect(get).toHaveBeenCalledTimes(3);
  expect(get.mock.calls[2][0]).toContain("limit=1");
  expect(get.mock.calls[2][0]).toContain("orderId=order-b");
  expect(result.current.photo?.thumbnailUrl).toBe("fresh");
  expect(result.current.photoRefreshError).toBeNull();
});

it("keeps refreshed media when an older page append completes later", async () => {
  login();
  const page = (orderId: string, offset: number, url: string) => ({
    items: [{ orderId, photoId: "photo-a" }], offset, limit: 1, total: 2,
    photo: { id: "photo-a", thumbnailUrl: url, mediaAvailable: true },
    event: { id: "event-a" }, selectedPurchase: null,
  });
  let resolveAppend!: (value: ReturnType<typeof page>) => void;
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(page("order-a", 0, "expired"))
    .mockImplementationOnce(() => new Promise((resolve) => { resolveAppend = resolve; }))
    .mockResolvedValueOnce({ ...page("order-a", 0, "fresh"), items: [] });
  const { result } = renderHook(() => usePhotographerPhotoPurchases("photo-a"), { wrapper });
  await waitFor(() => expect(result.current.photo?.thumbnailUrl).toBe("expired"));

  let append!: ReturnType<typeof result.current.fetchNextPage>;
  act(() => { append = result.current.fetchNextPage(); });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  const appendSignal = get.mock.calls[1][1]?.signal;
  await act(async () => { await result.current.refreshPhoto(); });
  expect(result.current.photo?.thumbnailUrl).toBe("fresh");

  await act(async () => { resolveAppend(page("order-b", 1, "expired")); await append; });

  expect(appendSignal?.aborted).toBe(true);
  expect(result.current.photo?.thumbnailUrl).toBe("fresh");
  expect(result.current.items.map((item) => item.orderId)).toEqual(["order-a"]);
});
