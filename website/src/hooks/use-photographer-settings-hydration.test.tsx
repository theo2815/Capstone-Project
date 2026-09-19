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
  brandName: "Brand A", brandColor: "none", bio: "", handle: "photo-a",
  regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "",
  watermarkUrl: "", watermarkSettings: null,
};

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
