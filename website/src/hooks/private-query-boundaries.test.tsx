import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { resetUserScopedStores } from "@/lib/auth-reset";
import type { Role } from "@/types/user";
import { useOrdersList, useOrderDetail } from "./use-orders";
import { useOwnedPhotoIds } from "./use-owned-photo-ids";
import { useSelfiesList } from "./use-selfies";
import { useAdminFlags, useAdminKpis, useAdminKpiTrend, useAdminSalesKpisLive, useAdminSalesByEventLive, useFlagActions } from "./use-admin-data";
import {
  usePhotographerEvents, usePhotographerEventDetail, usePhotographerEventPhotos,
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
afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

const page = { items: [{ id: "record-a" }], total: 1, offset: 0, limit: 24 };
describe.each([
  { name: "orders", role: "RUNNER", useRead: useOrdersList, payload: page },
  { name: "order detail", role: "RUNNER", useRead: () => useOrderDetail("order-a"), payload: { id: "order-a" } },
  { name: "selfies", role: "RUNNER", useRead: useSelfiesList, payload: [{ id: "selfie-a" }] },
  { name: "covered events", role: "PHOTOGRAPHER", useRead: usePhotographerEvents, payload: page },
  { name: "event detail", role: "PHOTOGRAPHER", useRead: () => usePhotographerEventDetail("event-a"), payload: { id: "event-a" } },
  { name: "event photos", role: "PHOTOGRAPHER", useRead: () => usePhotographerEventPhotos("event-a"), payload: page },
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
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "flags"] });
  });
  it.each([true, false])("suppresses stale completion (success=%s)", async (success) => {
    login("ADMIN");
    const pending = deferred<unknown>();
    vi.spyOn(api, "post").mockReturnValue(pending.promise);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(useFlagActions, { wrapper });
    const outcome = result.current[action]("flag-a", null);
    act(() => { useAuthStore.getState().logout(); login("ADMIN"); });
    if (success) pending.resolve({}); else pending.reject(new Error("offline"));
    await expect(outcome).resolves.toBe(false);
    expect(invalidate).not.toHaveBeenCalled();
  });
});
