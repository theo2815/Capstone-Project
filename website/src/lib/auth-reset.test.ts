import { afterEach, expect, it, vi } from "vitest";
import { captureGuestBuffer, clearLegacyPhotographerSettings, resetUserScopedStores } from "./auth-reset";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useSavedEventsStore } from "@/store/saved-events-store";

afterEach(() => { useAuthStore.getState().logout(); resetUserScopedStores(); });

it.each([false, true])("transfers only guest buffers across login (authenticated=%s)", (authenticated) => {
  useAuthStore.getState().logout();
  if (authenticated) useAuthStore.getState().setUser({ id: "old-owner", role: "RUNNER", name: "Runner", email: "runner@example.test", createdAt: "2026-09-10" });
  const item = { photoId: "old-photo", eventId: "event", thumbnailUrl: "", price: 100 };
  useCartStore.getState().setItems([item]);
  useSavedEventsStore.getState().setIds(["event"]);
  const restore = captureGuestBuffer();
  resetUserScopedStores();
  useAuthStore.getState().setUser({ id: "next-owner", role: "RUNNER", name: "Next", email: "next@example.test", createdAt: "2026-09-10" });
  restore();
  expect(useCartStore.getState().items).toEqual(authenticated ? [] : [item]);
  expect(useSavedEventsStore.getState().ids).toEqual(authenticated ? [] : ["event"]);
});

it("clears the pending payment from memory and persistence on a session reset", async () => {
  usePendingPaymentStore.getState().set({
    orderId: "order-1",
    imageUrl: "https://example.com/qr.png",
    expiresAt: "2026-09-10T12:00:00Z",
    returnToken: "guest-return-token",
    testUrl: null,
    email: "runner@example.com",
    total: 100,
    itemCount: 1,
    paidClaimedAt: "2026-09-10T11:45:00Z",
  });
  expect(localStorage.getItem("quickpitik-pending-payment")).toContain("order-1");

  resetUserScopedStores();

  expect(usePendingPaymentStore.getState().pending).toBeNull();
  await usePendingPaymentStore.persist.rehydrate();
  expect(usePendingPaymentStore.getState().pending).toBeNull();
  expect(localStorage.getItem("quickpitik-pending-payment")).not.toContain("guest-return-token");
});

it("removes the retired persisted admin submissions and decision log", () => {
  localStorage.setItem("quickpitik-admin-user-store", JSON.stringify({ state: { submissions: { private: {} }, log: [{ reason: "private" }] } }));
  resetUserScopedStores();
  expect(localStorage.getItem("quickpitik-admin-user-store")).toBeNull();
});

it("removes retired photographer settings on every reset without clearing unrelated storage", () => {
  localStorage.setItem("quickpitik-photographer-settings", JSON.stringify({ state: {
    brandName: "Private brand", socials: [{ url: "https://example.test/private" }],
    payouts: [{ accountHolderName: "Private Name", accountNumber: "1234567890", qrJson: "private-qr" }],
  } }));
  localStorage.setItem("unrelated-key", "keep");

  resetUserScopedStores();
  resetUserScopedStores();

  expect(localStorage.getItem("quickpitik-photographer-settings")).toBeNull();
  expect(localStorage.getItem("unrelated-key")).toBe("keep");
});

it("tolerates unavailable storage while removing retired photographer settings", () => {
  vi.stubGlobal("localStorage", undefined);
  try {
    expect(() => clearLegacyPhotographerSettings()).not.toThrow();
  } finally {
    vi.unstubAllGlobals();
  }

  const remove = vi.spyOn(localStorage, "removeItem").mockImplementation(() => { throw new Error("Storage blocked"); });
  try {
    expect(() => clearLegacyPhotographerSettings()).not.toThrow();
    expect(remove).toHaveBeenCalledWith("quickpitik-photographer-settings");
  } finally {
    remove.mockRestore();
  }
});
