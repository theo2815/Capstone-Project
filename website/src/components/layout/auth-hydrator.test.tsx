import { StrictMode } from "react";
import { act, cleanup, render, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as auth from "@/lib/auth";
import * as cartApi from "@/lib/api-cart";
import * as savedApi from "@/lib/api-saved-events";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import type { CartItem } from "@/types/order";
import type { User } from "@/types/user";
import { AuthHydrator } from "./auth-hydrator";

const runner: User = {
  id: "runner-a", role: "RUNNER", name: "Runner", email: "runner@example.com",
  createdAt: "2026-09-10T00:00:00Z",
};
const mergedItem: CartItem = {
  photoId: "server-photo", eventId: "event-1", thumbnailUrl: "/photo.jpg", price: 100,
};
const savedEvent: savedApi.SavedEventSummary = {
  id: "event-1", slug: "cebu-race", name: "Cebu Race", date: "2026-09-10",
  state: "live", bannerUrl: null, location: "Cebu", savedAt: "2026-09-10T00:00:00Z",
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

let client: QueryClient;
beforeEach(() => {
  resetUserScopedStores();
  useAuthStore.getState().setUser(runner);
  vi.spyOn(auth, "getAccessToken").mockReturnValue(null);
  vi.spyOn(auth, "getRefreshToken").mockReturnValue(null);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  resetUserScopedStores();
  vi.restoreAllMocks();
});

function startMerge() {
  const cart = deferred<CartItem[]>();
  const saved = deferred<savedApi.SavedEventSummary[]>();
  vi.spyOn(cartApi, "mergeCart").mockReturnValueOnce(cart.promise);
  vi.spyOn(savedApi, "mergeSavedEvents").mockReturnValueOnce(saved.promise);
  render(
    <StrictMode>
      <QueryClientProvider client={client}><AuthHydrator /></QueryClientProvider>
    </StrictMode>,
  );
  return { cart, saved };
}

it("merges guest buffers once and enables sync after confirmation for the same runner", async () => {
  const guestItems = [{ ...mergedItem, photoId: "guest-photo" }];
  useCartStore.getState().setItems(guestItems);
  useSavedEventsStore.getState().setIds(["guest-event"]);
  const { cart, saved } = startMerge();

  expect(cartApi.mergeCart).toHaveBeenCalledExactlyOnceWith(guestItems);
  expect(savedApi.mergeSavedEvents).toHaveBeenCalledExactlyOnceWith(["guest-event"]);
  expect(useCartStore.getState().syncEnabled).toBe(false);
  await act(async () => { cart.resolve([mergedItem]); });
  expect(useCartStore.getState().items).toEqual(guestItems);

  await act(async () => { saved.resolve([savedEvent]); });
  expect(useCartStore.getState()).toMatchObject({ items: [mergedItem], syncEnabled: true });
  expect(useSavedEventsStore.getState()).toMatchObject({ ids: [savedEvent.id], summaries: [savedEvent], syncEnabled: true });
});

it.each(["success", "failure"])("discards a guest merge completing after logout (%s)", async (outcome) => {
  const { cart, saved } = startMerge();
  act(() => {
    useAuthStore.getState().logout();
    resetUserScopedStores();
    // A new guest's activity must also survive the old session's response.
    useCartStore.getState().setItems([{ ...mergedItem, photoId: "new-guest-photo" }]);
    useSavedEventsStore.getState().setIds(["new-guest-event"]);
  });
  const guestCart = useCartStore.getState();
  const guestSaved = useSavedEventsStore.getState();

  await act(async () => {
    if (outcome === "success") cart.resolve([mergedItem]);
    else cart.reject(new Error("Merge failed"));
    saved.resolve([savedEvent]);
  });

  expect(useCartStore.getState()).toBe(guestCart);
  expect(useSavedEventsStore.getState()).toBe(guestSaved);
  expect(useCartStore.getState().syncEnabled).toBe(false);
  expect(useSavedEventsStore.getState().syncEnabled).toBe(false);
});

it("does not overwrite a different runner's merged data with an older response", async () => {
  const { cart, saved } = startMerge();
  const nextItem = { ...mergedItem, photoId: "runner-b-photo" };
  const nextEvent = { ...savedEvent, id: "runner-b-event" };
  vi.mocked(cartApi.mergeCart).mockResolvedValueOnce([nextItem]);
  vi.mocked(savedApi.mergeSavedEvents).mockResolvedValueOnce([nextEvent]);
  act(() => {
    resetUserScopedStores();
    useAuthStore.getState().setUser({ ...runner, id: "runner-b" });
  });
  await waitFor(() => expect(useCartStore.getState().syncEnabled).toBe(true));
  const nextCart = useCartStore.getState();
  const nextSaved = useSavedEventsStore.getState();

  await act(async () => { cart.resolve([mergedItem]); saved.resolve([savedEvent]); });

  expect(useCartStore.getState()).toBe(nextCart);
  expect(useSavedEventsStore.getState()).toBe(nextSaved);
  expect(nextCart.items).toEqual([nextItem]);
  expect(nextSaved.ids).toEqual([nextEvent.id]);
});

it("starts a fresh merge on same-runner relogin and ignores the previous session", async () => {
  const { cart, saved } = startMerge();
  const latest = deferred<CartItem[]>();
  vi.mocked(cartApi.mergeCart).mockReturnValueOnce(latest.promise);
  vi.mocked(savedApi.mergeSavedEvents).mockResolvedValueOnce([]);
  act(() => {
    useAuthStore.getState().logout(); resetUserScopedStores();
    useAuthStore.getState().setUser(runner);
  });
  expect(cartApi.mergeCart).toHaveBeenCalledTimes(2);
  await act(async () => { cart.resolve([mergedItem]); saved.resolve([savedEvent]); });
  expect(useCartStore.getState()).toMatchObject({ items: [], syncEnabled: false });
  await act(async () => { latest.resolve([{ ...mergedItem, price: 200 }]); });
  expect(useCartStore.getState()).toMatchObject({ items: [{ ...mergedItem, price: 200 }], syncEnabled: true });
});

it.each([true, false])("ignores initial auth hydration after a new login (old response success=%s)", async (success) => {
  const pending = deferred<Response>();
  useAuthStore.getState().logout();
  vi.mocked(auth.getAccessToken).mockReturnValue("old-access");
  vi.mocked(auth.getRefreshToken).mockReturnValue("old-refresh");
  vi.stubGlobal("fetch", vi.fn(() => pending.promise));
  const clearTokens = vi.spyOn(auth, "clearTokens");
  vi.spyOn(cartApi, "mergeCart").mockResolvedValue([]); vi.spyOn(savedApi, "mergeSavedEvents").mockResolvedValue([]);
  render(<QueryClientProvider client={client}><AuthHydrator /></QueryClientProvider>);
  act(() => useAuthStore.getState().setUser({ ...runner, id: "new-runner" }));
  await act(async () => { pending.resolve(new Response(JSON.stringify({ success, data: runner }), { status: success ? 200 : 401 })); });
  expect(useAuthStore.getState().user?.id).toBe("new-runner");
  expect(clearTokens).not.toHaveBeenCalled();
  expect(fetch).toHaveBeenCalledTimes(1); // No old-session refresh either.
  vi.unstubAllGlobals();
});

it("restores a valid current session on page refresh", async () => {
  useAuthStore.getState().logout();
  vi.mocked(auth.getAccessToken).mockReturnValue("access");
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ success: true, data: runner }))));
  vi.spyOn(cartApi, "mergeCart").mockResolvedValue([mergedItem]); vi.spyOn(savedApi, "mergeSavedEvents").mockResolvedValue([savedEvent]);
  render(<QueryClientProvider client={client}><AuthHydrator /></QueryClientProvider>);
  await waitFor(() => expect(useCartStore.getState().syncEnabled).toBe(true));
  expect(useAuthStore.getState().user).toEqual(runner);
  expect(useSavedEventsStore.getState().ids).toEqual([savedEvent.id]);
  vi.unstubAllGlobals();
});
