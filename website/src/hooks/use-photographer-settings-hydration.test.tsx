import { StrictMode } from "react";
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { usePhotographerSettingsHydration } from "./use-photographer-settings-hydration";
const user = { id: "photo-a", role: "PHOTOGRAPHER" as const, name: "A", email: "a@example.test", createdAt: "2026-09-10" };
const brand = { brandName: "Brand A", brandColor: "none", bio: "", handle: "photo-a", regionCode: "", provinceCode: "", avatarUrl: null };
beforeEach(() => { resetUserScopedStores(); useAuthStore.getState().setUser(user); });
afterEach(() => { cleanup(); useAuthStore.getState().logout(); resetUserScopedStores(); vi.restoreAllMocks(); vi.useRealTimers(); });

it("deduplicates StrictMode/remount hydration and drops all old-account blocks", async () => {
  let finish!: (value: unknown) => void;
  const pending = new Promise((resolve) => { finish = resolve; });
  vi.spyOn(api, "get").mockImplementation(() => pending);
  const hook = renderHook(usePhotographerSettingsHydration, { wrapper: StrictMode });
  expect(api.get).toHaveBeenCalledTimes(3);
  hook.unmount();
  useAuthStore.getState().logout(); resetUserScopedStores(); useAuthStore.getState().setUser({ ...user, id: "photo-b" });
  vi.mocked(api.get).mockImplementation((path) => Promise.resolve(path.endsWith("/brand") ? { ...brand, brandName: "Brand B" } : []));
  renderHook(usePhotographerSettingsHydration);
  await act(async () => {});
  expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand B");
  await act(async () => { finish(brand); });
  expect(usePhotographerSettingsStore.getState()).toMatchObject({ brandName: "Brand B", socials: [], payouts: [] });
});

it("retries only a failed block, preserves edits and stops after the retry budget", async () => {
  vi.useFakeTimers(); vi.spyOn(console, "warn").mockImplementation(() => {});
  const get = vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.endsWith("/payouts")) return Promise.reject(new Error("Offline"));
    return Promise.resolve(path.endsWith("/brand") ? brand : []);
  });
  const hook = renderHook(usePhotographerSettingsHydration, { wrapper: StrictMode });
  await act(async () => {});
  act(() => usePhotographerSettingsStore.setState({ brandName: "Unsaved edit" }));
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  await act(async () => { await vi.advanceTimersByTimeAsync(6000); });
  expect(get.mock.calls.filter(([path]) => path.endsWith("/brand"))).toHaveLength(1);
  expect(get.mock.calls.filter(([path]) => path.endsWith("/payouts"))).toHaveLength(3);
  expect(usePhotographerSettingsStore.getState().brandName).toBe("Unsaved edit");
  hook.unmount(); renderHook(usePhotographerSettingsHydration); await act(async () => {});
  expect(get).toHaveBeenCalledTimes(5);
});

it("cancels old retries and lets a same-account relogin hydrate afresh", async () => {
  vi.useFakeTimers(); vi.spyOn(console, "warn").mockImplementation(() => {});
  vi.spyOn(api, "get").mockRejectedValue(new Error("Offline"));
  renderHook(usePhotographerSettingsHydration); await act(async () => {});
  act(() => { useAuthStore.getState().logout(); resetUserScopedStores(); });
  await act(async () => { await vi.advanceTimersByTimeAsync(2000); });
  expect(api.get).toHaveBeenCalledTimes(3);
  vi.mocked(api.get).mockImplementation((path) => Promise.resolve(path.endsWith("/brand") ? brand : []));
  act(() => useAuthStore.getState().setUser(user)); await act(async () => {});
  expect(api.get).toHaveBeenCalledTimes(6);
  expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand A");
});
