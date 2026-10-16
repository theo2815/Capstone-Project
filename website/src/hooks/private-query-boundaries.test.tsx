import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { onlineManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { resetUserScopedStores } from "@/lib/auth-reset";
import type { Role } from "@/types/user";
import { useOrdersList, useOrderDetail } from "./use-orders";
import { useOwnedPhotoIds } from "./use-owned-photo-ids";
import { useSelfiesList } from "./use-selfies";
import { usePhotographerSettingsQueries } from "./use-photographer-settings";
import { usePhotographerPurchasedPhotos, usePhotographerPhotoPurchases } from "./use-photographer-purchases";
import { useAdminFlags, useAdminKpis, useAdminKpiTrend, useAdminSalesKpisLive, useAdminSalesByEventLive, useFlagActions } from "./use-admin-data";
import {
  usePhotographerEvents, usePhotographerEventDetail, usePhotographerEventPhotos, usePhotographerEventMetadataVisibility,
  usePhotographerEarnings, usePhotographerPerEventEarnings, usePhotographerPayouts,
  usePhotographerPayoutBalance, usePhotographerTransactions, usePublicPhotographer,
} from "./use-photographer-data";

vi.mock("@/lib/constants", async (original) => ({
  ...await original<typeof import("@/lib/constants")>(), ADMIN_FLAGS_ENABLED: true,
}));

let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
function login(role: Role, id = "owner-a") {
  useAuthStore.getState().setUser({ id, role, name: id, email: `${id}@example.test`, createdAt: "2026-09-10" });
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  useAuthStore.getState().logout();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); onlineManager.setOnline(true); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

const page = { items: [{ id: "record-a" }], total: 1, offset: 0, limit: 24 };

const offlineLists = [
  { name: "Purchased", useRead: () => usePhotographerPurchasedPhotos("event-a", true), payload: { ...page, items: [{ photo: { id: "photo-a" }, purchaseCount: 1 }] } },
  { name: "receipts", useRead: () => usePhotographerPhotoPurchases("photo-a", "order-a"), payload: { ...page, items: [{ orderId: "order-a", photoId: "photo-a" }], photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: { orderId: "order-a", photoId: "photo-a" } } },
  { name: "event earnings", useRead: usePhotographerPerEventEarnings, payload: { ...page, items: [{ eventId: "event-a" }] } },
  { name: "transactions", useRead: usePhotographerTransactions, payload: { ...page, monthTotals: { "2026-10": 75 } } },
] as const;

it.each(offlineLists)("$name exposes initial and append pause, recovers and cancels old-session queued work", async ({ useRead, payload }) => {
  login("PHOTOGRAPHER");
  onlineManager.setOnline(false);
  const get = vi.spyOn(api, "get").mockResolvedValue({ ...payload, total: 100 });
  const { result } = renderHook(() => useRead(), { wrapper });
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", true));
  expect(result.current.error).toBeNull();
  expect(result.current.items).toEqual([]);
  expect(get).not.toHaveBeenCalled();
  act(() => onlineManager.setOnline(true));
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  const loaded = result.current.items;
  act(() => { onlineManager.setOnline(false); void result.current.fetchNextPage(); });
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", true));
  expect(result.current.items).toEqual(loaded);
  expect(result.current.appendError).toBeNull();
  expect(get).toHaveBeenCalledTimes(1);
  act(() => onlineManager.setOnline(true));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", false));
  expect(result.current.items).toEqual(loaded);
  act(() => { onlineManager.setOnline(false); void result.current.fetchNextPage(); });
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", true));
  act(() => { useAuthStore.getState().logout(); client.clear(); login("RUNNER"); onlineManager.setOnline(true); });
  await waitFor(() => expect(result.current.items).toEqual([]));
  expect(get).toHaveBeenCalledTimes(2);
});

it.each(offlineLists)("$name exposes append pause with loaded rows intact", async ({ useRead, payload }) => {
  login("PHOTOGRAPHER");
  vi.spyOn(api, "get").mockResolvedValue({ ...payload, total: 100 });
  const { result } = renderHook(() => useRead(), { wrapper });
  await waitFor(() => expect(result.current.items).toHaveLength(1));
  const loaded = result.current.items;
  act(() => { onlineManager.setOnline(false); void result.current.fetchNextPage(); });
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", true));
  expect(result.current.items).toEqual(loaded);
  expect(result.current.appendError).toBeNull();
});

it("uncached event detail exposes pause without missing or error and recovers on reconnect", async () => {
  login("PHOTOGRAPHER");
  onlineManager.setOnline(false);
  const get = vi.spyOn(api, "get").mockResolvedValue({ id: "event-a" });
  const { result } = renderHook(() => usePhotographerEventDetail("event-a", { includeRemoved: true }), { wrapper });
  await waitFor(() => expect(result.current).toHaveProperty("isPaused", true));
  expect(result.current.isMissing).toBe(false);
  expect(result.current.error).toBeNull();
  expect(get).not.toHaveBeenCalled();
  act(() => onlineManager.setOnline(true));
  await waitFor(() => expect(result.current.detail?.id).toBe("event-a"));
  expect(result.current).toHaveProperty("isPaused", false);
});

it("deduplicates event earnings while retaining raw offsets, totals and rows through append retry and refresh", async () => {
  login("PHOTOGRAPHER");
  const earning = (eventId: string, revenueKept: number) => ({
    eventId, eventName: eventId, eventDate: "2026-09-01", photoCount: 2,
    salesCount: 1, purchaseCount: 1, revenueKept,
  });
  const e1 = earning("e1", 40);
  const e2 = earning("e2", 0);
  const e3 = earning("e3", 0);
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce({ items: [e1, e2], total: 25, offset: 0, limit: 7 })
    .mockRejectedValueOnce(new ApiError([{ code: "RATE_LIMITED", message: "Try again" }], 429))
    .mockResolvedValueOnce({ items: [earning("e2", 99), e3], total: 25, offset: 7, limit: 6 })
    .mockResolvedValueOnce({ items: [], total: 25, offset: 13, limit: 10 })
    .mockResolvedValueOnce({ items: [e3, e1], total: 2, offset: 0, limit: 10 });
  const { result } = renderHook(usePhotographerPerEventEarnings, { wrapper });
  await waitFor(() => expect(result.current.items).toEqual([e1, e2]));
  act(() => result.current.fetchNextPage());
  await waitFor(() => expect(result.current.appendError).toBeInstanceOf(ApiError));
  expect(result.current.items).toEqual([e1, e2]);
  expect(result.current.total).toBe(25);
  act(() => result.current.fetchNextPage());
  await waitFor(() => expect(result.current.items).toEqual([e1, e2, e3]));
  expect(result.current.appendError).toBeNull();
  expect(result.current.total).toBe(25);
  act(() => result.current.fetchNextPage());
  await waitFor(() => expect(result.current.hasNextPage).toBe(false));
  expect(result.current.items).toEqual([e1, e2, e3]);
  expect(result.current.total).toBe(25);
  act(() => result.current.refetch());
  await waitFor(() => expect(result.current.items).toEqual([e3, e1]));
  expect(result.current.total).toBe(2);
  expect(get.mock.calls.map(([path]) => new URLSearchParams(path.split("?")[1]).get("offset")))
    .toEqual(["0", "7", "7", "13", "0"]);
});
describe.each([
  { name: "orders", role: "RUNNER", useRead: useOrdersList, payload: page },
  { name: "order detail", role: "RUNNER", useRead: () => useOrderDetail("order-a"), payload: { id: "order-a" } },
  { name: "selfies", role: "RUNNER", useRead: useSelfiesList, payload: [{ id: "selfie-a" }] },
  { name: "covered events", role: "PHOTOGRAPHER", useRead: usePhotographerEvents, payload: page },
  { name: "event detail", role: "PHOTOGRAPHER", useRead: () => usePhotographerEventDetail("event-a"), payload: { id: "event-a" } },
  { name: "event photos", role: "PHOTOGRAPHER", useRead: () => usePhotographerEventPhotos("event-a"), payload: page },
  { name: "event metadata visibility", role: "PHOTOGRAPHER", useRead: () => usePhotographerEventMetadataVisibility("event-a"), payload: { metadataPublic: false } },
  { name: "earnings", role: "PHOTOGRAPHER", useRead: usePhotographerEarnings, payload: { total: 100 } },
  { name: "per-event earnings", role: "PHOTOGRAPHER", useRead: usePhotographerPerEventEarnings, payload: page },
  { name: "payouts", role: "PHOTOGRAPHER", useRead: usePhotographerPayouts, payload: page },
  { name: "payout balance", role: "PHOTOGRAPHER", useRead: usePhotographerPayoutBalance, payload: { unpaid: 100 } },
  { name: "transactions", role: "PHOTOGRAPHER", useRead: usePhotographerTransactions, payload: { ...page, monthTotals: {} } },
  { name: "admin KPIs", role: "ADMIN", useRead: useAdminKpis, payload: { total: 100 } },
  { name: "admin flags", role: "ADMIN", useRead: useAdminFlags, payload: page },
  { name: "admin trend", role: "ADMIN", useRead: useAdminKpiTrend, payload: [] },
  { name: "admin sales", role: "ADMIN", useRead: useAdminSalesKpisLive, payload: { total: 100 } },
  { name: "admin event sales", role: "ADMIN", useRead: useAdminSalesByEventLive, payload: page },
] as const)("$name session boundary", ({ role, useRead, payload }) => {
  it("gates guests and discards pending reads across logout and same-account relogin", async () => {
    const old = deferred<unknown>();
    const get = vi.spyOn(api, "get").mockReturnValueOnce(old.promise).mockResolvedValue(payload);
    const hook = renderHook(() => useRead(), { wrapper });
    expect(get).not.toHaveBeenCalled();
    act(() => login(role));
    await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
    const signal = get.mock.calls[0][1]?.signal;
    expect(signal).toBeInstanceOf(AbortSignal);
    act(() => { useAuthStore.getState().logout(); client.clear(); });
    expect(signal?.aborted).toBe(true);
    act(() => login(role));
    await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(client.isFetching()).toBe(0));
    const current = hook.result.current;
    await act(async () => old.resolve(payload));
    expect(hook.result.current).toEqual(current);
    expect(client.getQueryCache().getAll().filter((q) => q.state.data !== undefined)).toHaveLength(1);
  });
  if (role !== "RUNNER") {
    it("does not request another role's private endpoint", () => {
      login("RUNNER");
      const get = vi.spyOn(api, "get");
      renderHook(() => useRead(), { wrapper });
      expect(get).not.toHaveBeenCalled();
    });
  }
});

it.each(["purchased photos", "photo purchases"])("%s seller session boundary", async (name) => {
  const payload = name === "purchased photos" ? page : { ...page, photo: { id: "photo-a" }, event: { id: "event-a" }, selectedPurchase: null };
  login("RUNNER");
  const get = vi.spyOn(api, "get").mockResolvedValue(payload);
  const hook = renderHook(() => name === "purchased photos"
    ? usePhotographerPurchasedPhotos("event-a", true)
    : usePhotographerPhotoPurchases("photo-a", "order-a"), { wrapper });
  expect(get).not.toHaveBeenCalled();
  act(() => login("PHOTOGRAPHER"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  expect(get.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
  expect(client.getQueryCache().getAll().some((q) => q.queryKey.includes("owner-a") && q.queryKey.includes("PHOTOGRAPHER") && q.queryKey.includes(name === "purchased photos" ? "event-a" : "photo-a"))).toBe(true);
  hook.unmount();
});

it("keeps a late metadata settings read out of a new photographer session", async () => {
  const old = deferred<{ metadataPublicByDefault: boolean }>();
  let metadataReads = 0;
  const get = vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.endsWith("/metadata-settings")) {
      metadataReads += 1;
      return metadataReads === 1 ? old.promise : Promise.resolve({ metadataPublicByDefault: true });
    }
    return Promise.resolve(path.endsWith("/brand") ? {} : []);
  });
  const hook = renderHook(usePhotographerSettingsQueries, { wrapper });
  expect(get).not.toHaveBeenCalled();
  act(() => login("PHOTOGRAPHER"));
  await waitFor(() => expect(metadataReads).toBe(1));
  const signal = get.mock.calls.find(([path]) => path.endsWith("/metadata-settings"))?.[1]?.signal;
  expect(signal).toBeInstanceOf(AbortSignal);
  act(() => { useAuthStore.getState().logout(); client.clear(); login("PHOTOGRAPHER"); });
  await waitFor(() => expect(hook.result.current.metadata.data).toEqual({ metadataPublicByDefault: true }));
  expect(signal?.aborted).toBe(true);
  await act(async () => old.resolve({ metadataPublicByDefault: false }));
  expect(hook.result.current.metadata.data).toEqual({ metadataPublicByDefault: true });
});

it("shares orders between receipts and owned-photo hints; invalidation refreshes both", async () => {
  login("RUNNER");
  const get = vi.spyOn(api, "get").mockResolvedValue({ ...page, items: [
    { id: "paid", status: "PAID", photoIds: ["owned"] },
    { id: "pending", status: "PENDING", photoIds: ["pending"] },
    { id: "refunded", status: "REFUNDED", photoIds: ["refunded"] },
    { id: "legacy", photoIds: ["unconfirmed"] },
  ] });
  const { result } = renderHook(() => ({ orders: useOrdersList(), owned: useOwnedPhotoIds() }), { wrapper });
  await waitFor(() => expect(result.current.owned).toEqual(new Set(["owned"])));
  expect(get).toHaveBeenCalledTimes(1);
  get.mockResolvedValue({ ...page, items: [{ id: "new", status: "FULFILLED", photoIds: ["new-photo"] }] });
  await act(async () => { await client.invalidateQueries({ queryKey: ["me", "orders"] }); });
  await waitFor(() => expect(result.current.owned).toEqual(new Set(["new-photo"])));
  expect(result.current.orders.orders[0].id).toBe("new");
});

it("does not reuse cached orders after an account switch even before cache clearing", async () => {
  login("RUNNER");
  const next = deferred<unknown>();
  vi.spyOn(api, "get").mockResolvedValueOnce({ ...page, items: [{ id: "a", status: "PAID", photoIds: ["a"] }] }).mockReturnValueOnce(next.promise);
  const { result } = renderHook(useOwnedPhotoIds, { wrapper });
  await waitFor(() => expect(result.current.has("a")).toBe(true));
  act(() => login("RUNNER", "owner-b"));
  expect(result.current.size).toBe(0);
  await act(async () => next.resolve({ ...page, items: [{ id: "b", status: "PAID", photoIds: ["b"] }] }));
  await waitFor(() => expect(result.current).toEqual(new Set(["b"])));
});

it("drains retired persisted orders without granting guest ownership", () => {
  localStorage.setItem("quickpitik-orders", JSON.stringify({ state: { orders: [{ photoIds: ["old-photo"] }] } }));
  const get = vi.spyOn(api, "get");
  const { result } = renderHook(useOwnedPhotoIds, { wrapper });
  expect(result.current.size).toBe(0);
  expect(get).not.toHaveBeenCalled();
  resetUserScopedStores();
  expect(localStorage.getItem("quickpitik-orders")).toBeNull();
});

it("keeps public photographer profiles available to guests", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ handle: "public-person" });
  renderHook(() => usePublicPhotographer("public-person"), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  expect(client.getQueryCache().getAll()[0].queryKey).toEqual(["photographer", "public", "public-person"]);
});

describe.each(["hide", "dismiss", "escalate"] as const)("flag %s", (action) => {
  it("retains cached data on failure and allows retry after backend confirmation", async () => {
    login("ADMIN");
    const post = vi.spyOn(api, "post").mockRejectedValueOnce(new Error("offline")).mockResolvedValue({});
    client.setQueryData(["admin", "flags", "fixture"], page);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(useFlagActions, { wrapper });
    await expect(result.current[action]("flag-a", null)).rejects.toThrow("offline");
    expect(client.getQueryData(["admin", "flags", "fixture"])).toEqual(page);
    expect(invalidate).not.toHaveBeenCalled();
    await expect(result.current[action]("flag-a", null)).resolves.toBe(true);
    expect(post).toHaveBeenCalledTimes(2);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "flags"] }, { throwOnError: true });
  });
  it.each([true, false])("suppresses stale completion (success=%s)", async (success) => {
    login("ADMIN");
    const pending = deferred<unknown>();
    vi.spyOn(api, "post").mockReturnValue(pending.promise);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(useFlagActions, { wrapper });
    const outcome = result.current[action]("flag-a", null);
    await waitFor(() => expect(api.post).toHaveBeenCalledTimes(1));
    act(() => { useAuthStore.getState().logout(); login("ADMIN"); });
    if (success) pending.resolve({}); else pending.reject(new Error("offline"));
    await expect(outcome).resolves.toBe(false);
    expect(invalidate).not.toHaveBeenCalled();
  });
});
