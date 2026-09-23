import { act } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { useAuthStore } from "./auth-store";
import { useCartStore } from "./cart-store";
import { useSavedEventsStore } from "./saved-events-store";

const runner = { id: "runner", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-09-10" };
const item = { photoId: "photo", eventId: "event", thumbnailUrl: "/photo.jpg", price: 100 };
const event = { id: "event", slug: "race", name: "Race", date: "2026-09-10", state: "live" as const, bannerUrl: null, location: "Cebu", savedAt: "2026-09-10" };
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}
function logout() { useAuthStore.getState().logout(); resetUserScopedStores(); }
beforeEach(() => {
  logout(); useAuthStore.getState().setUser(runner);
  useCartStore.getState().setSyncEnabled(true);
  useSavedEventsStore.getState().setSyncEnabled(true);
});
afterEach(() => { logout(); vi.restoreAllMocks(); });

it.each(["guest", "different account", "same account"])("discards late add reconciliation and rollback in a new %s session", async (next) => {
  const cart = deferred<unknown>(), saved = deferred<unknown>();
  vi.spyOn(api, "post").mockImplementation((path) => path.includes("/cart/") ? cart.promise : saved.promise);
  useCartStore.getState().addItem(item);
  useSavedEventsStore.getState().save(event.id, event);
  logout();
  if (next !== "guest") useAuthStore.getState().setUser({ ...runner, id: next === "same account" ? runner.id : "other" });
  // Reusing the same IDs makes an old rollback just as dangerous as an old success.
  useCartStore.getState().setItems([{ ...item, price: 200 }]);
  useSavedEventsStore.getState().setSummaries([{ ...event, name: "New session" }]);
  const cartState = useCartStore.getState(), savedState = useSavedEventsStore.getState();
  cart.resolve({ ...item, price: 500 }); saved.reject(new Error("Late failure"));
  await Promise.allSettled([cart.promise, saved.promise]);
  await act(async () => {});
  expect(useCartStore.getState()).toBe(cartState);
  expect(useSavedEventsStore.getState()).toBe(savedState);
});

it("ignores stale bookmark success and cart add failure after logout", async () => {
  const cart = deferred<unknown>(), saved = deferred<unknown>();
  vi.spyOn(api, "post").mockImplementation((path) => path.includes("/cart/") ? cart.promise : saved.promise);
  useCartStore.getState().addItem(item); useSavedEventsStore.getState().save(event.id);
  logout(); useCartStore.getState().addItem(item);
  cart.reject(new Error("Late failure")); saved.resolve(event);
  await Promise.allSettled([cart.promise, saved.promise]);
  await act(async () => {});
  expect(useCartStore.getState().items).toEqual([item]);
  expect(useSavedEventsStore.getState().summaries).toEqual([]);
});

it.each([false, true])("only rolls back failed removals in their original session (logout=%s)", async (endsSession) => {
  useCartStore.getState().setItems([item]); useSavedEventsStore.getState().setSummaries([event]);
  const pending = deferred<unknown>(); vi.spyOn(api, "delete").mockReturnValue(pending.promise);
  useCartStore.getState().removeItem(item.photoId); useSavedEventsStore.getState().unsave(event.id);
  if (endsSession) logout();
  pending.reject(new Error("Delete failed"));
  await Promise.allSettled([pending.promise]); await act(async () => {});
  expect(useCartStore.getState().items).toEqual(endsSession ? [] : [item]);
  expect(useSavedEventsStore.getState().summaries).toEqual(endsSession ? [] : [event]);
});

it("keeps current-session reconciliation and failed-add retry working", async () => {
  vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Offline")).mockResolvedValueOnce({ ...item, price: 120 }).mockResolvedValueOnce(event);
  useCartStore.getState().addItem(item);
  await act(async () => {});
  expect(useCartStore.getState().items).toEqual([]);
  useCartStore.getState().addItem(item); useSavedEventsStore.getState().save(event.id);
  await act(async () => {});
  expect(useCartStore.getState().items[0].price).toBe(120);
  expect(useSavedEventsStore.getState().summaries).toEqual([event]);
});
