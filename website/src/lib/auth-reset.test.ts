import { expect, it } from "vitest";
import { resetUserScopedStores } from "./auth-reset";
import { usePendingPaymentStore } from "@/store/pending-payment-store";

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
