import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { usePhotographerSettingsHydration } from "./use-photographer-settings-hydration";

const user = { id: "photo-a", role: "PHOTOGRAPHER" as const, name: "A", email: "a@example.test", createdAt: "2026-09-10" };
const brand = {
  brandName: "Brand A", brandColor: "indigo", bio: "Existing bio", handle: "photo-a",
  regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "",
  watermarkUrl: "", watermarkSettings: null,
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  resetUserScopedStores();
  useAuthStore.getState().setUser(user);
});
afterEach(() => {
  cleanup();
  useAuthStore.getState().logout();
  resetUserScopedStores();
  vi.restoreAllMocks();
});

it("hydrates the draft bridge from session-scoped Query data", async () => {
  vi.spyOn(api, "get").mockImplementation((path) =>
    Promise.resolve(path.endsWith("/brand") ? brand : []),
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand A"));
  expect(api.get).toHaveBeenCalledTimes(3);
});

it.each([
  ["brand", "socials", "payouts"],
  ["brand", "payouts", "socials"],
  ["socials", "brand", "payouts"],
  ["socials", "payouts", "brand"],
  ["payouts", "socials", "brand"],
  ["payouts", "brand", "socials"],
] as const)("waits for all initial reads before applying %s → %s → %s", async (...order) => {
  const reads = {
    brand: deferred<typeof brand>(),
    socials: deferred<[]>(),
    payouts: deferred<[]>(),
  };
  vi.spyOn(api, "get").mockImplementation((path) =>
    path.endsWith("/brand") ? reads.brand.promise
      : path.endsWith("/socials") ? reads.socials.promise
        : reads.payouts.promise,
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });

  const resolveRead = (key: (typeof order)[number]) => {
    if (key === "brand") reads.brand.resolve(brand);
    else reads[key].resolve([]);
  };

  for (const key of order.slice(0, -1)) {
    await act(async () => { resolveRead(key); });
    expect(usePhotographerSettingsStore.getState().brandName).toBe("");
    expect(usePhotographerSettingsStore.getState().hydratedSessionVersion).toBeNull();
  }

  const last = order.at(-1)!;
  await act(async () => { resolveRead(last); });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand A"));
  expect(usePhotographerSettingsStore.getState().hydratedSessionVersion)
    .toBe(useAuthStore.getState().sessionVersion);
});

it("does not reuse readiness or identical response objects after same-account login", async () => {
  const socials: [] = [];
  const payouts: [] = [];
  vi.spyOn(api, "get").mockImplementation((path) =>
    Promise.resolve(path.endsWith("/brand") ? brand : path.endsWith("/socials") ? socials : payouts),
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand A"));

  act(() => {
    useAuthStore.getState().logout();
    resetUserScopedStores();
    useAuthStore.getState().setUser(user);
  });

  const nextVersion = useAuthStore.getState().sessionVersion;
  await waitFor(() => expect(usePhotographerSettingsStore.getState().hydratedSessionVersion).toBe(nextVersion));
  expect(usePhotographerSettingsStore.getState()).toMatchObject({
    brandName: "Brand A", brandColor: "indigo", bio: "Existing bio",
  });
});

it("keeps existing drafts when an individual server read fails", async () => {
  usePhotographerSettingsStore.setState({ brandName: "Unsaved edit" });
  vi.spyOn(api, "get").mockImplementation((path) =>
    path.endsWith("/brand") ? Promise.reject(new Error("Offline")) : Promise.resolve([]),
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });
  await act(async () => {});
  expect(usePhotographerSettingsStore.getState().brandName).toBe("Unsaved edit");
});

it("does not apply a late response after logout", async () => {
  let finish!: (value: unknown) => void;
  const pending = new Promise((resolve) => { finish = resolve; });
  vi.spyOn(api, "get").mockImplementation((path) =>
    path.endsWith("/brand") ? pending : Promise.resolve([]),
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });
  act(() => {
    useAuthStore.getState().logout();
    resetUserScopedStores();
  });
  await act(async () => { finish(brand); });
  expect(usePhotographerSettingsStore.getState().brandName).toBe("");
});

it("does not apply an old account response after the replacement account is ready", async () => {
  const oldBrand = deferred<typeof brand>();
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (!path.endsWith("/brand")) return Promise.resolve([]);
    return useAuthStore.getState().user?.id === "photo-a"
      ? oldBrand.promise
      : Promise.resolve({ ...brand, brandName: "Brand B", bio: "B bio" });
  });
  renderHook(usePhotographerSettingsHydration, { wrapper });

  act(() => {
    useAuthStore.getState().logout();
    resetUserScopedStores();
    useAuthStore.getState().setUser({ ...user, id: "photo-b" });
  });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand B"));
  await act(async () => { oldBrand.resolve(brand); });

  expect(usePhotographerSettingsStore.getState()).toMatchObject({
    brandName: "Brand B", bio: "B bio",
    hydratedSessionVersion: useAuthStore.getState().sessionVersion,
  });
});

it("releases draft ownership when switching photographer accounts", async () => {
  vi.spyOn(api, "get").mockImplementation((path) =>
    Promise.resolve(path.endsWith("/brand")
      ? { ...brand, brandName: useAuthStore.getState().user?.id === "photo-b" ? "Brand B" : "Brand A" }
      : []),
  );
  renderHook(usePhotographerSettingsHydration, { wrapper });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand A"));
  act(() => {
    usePhotographerSettingsStore.getState().beginDraft();
    usePhotographerSettingsStore.getState().setBrandName("Unsaved A");
    useAuthStore.getState().logout();
    resetUserScopedStores();
    useAuthStore.getState().setUser({ ...user, id: "photo-b" });
  });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().brandName).toBe("Brand B"));
  expect(usePhotographerSettingsStore.getState().draftEditing).toBe(false);
});
