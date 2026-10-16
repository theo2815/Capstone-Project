import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { usePendingPaymentStore, type PendingPayment } from "@/store/pending-payment-store";
import type { Order } from "@/types/order";
import { useCheckoutWorkflow } from "./use-checkout-workflow";

let client: QueryClient;
let nextOrder: Order;
let live: PendingPayment;

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  sessionStorage.clear();
  useAuthStore.getState().logout();
  useCartStore.setState({ items: [
    { photoId: "photo-1", eventId: "event-1", price: 125, thumbnailUrl: "" },
  ] });
  live = {
    orderId: "order-1", imageUrl: "data:image/png;base64,cXI=",
    expiresAt: new Date(Date.now() + 600_000).toISOString(),
    returnToken: "guest-token", testUrl: null, email: "runner@example.test",
    total: 125, totalSource: "checkout", itemCount: 1, paidClaimedAt: null,
  };
  usePendingPaymentStore.getState().clear();
  nextOrder = {
    id: live.orderId, status: "PENDING", items: [], totalAmount: 125, checkoutTotalAmount: 125,
    paymentMethod: "qrph", createdAt: new Date().toISOString(), qrPh: live,
  };
  vi.spyOn(api, "get").mockResolvedValue({ id: live.orderId, status: "PENDING", paidAt: null });
  vi.spyOn(api, "post").mockImplementation(async (path) => {
    if (path === "/coupons/preview") {
      return { code: null, discountTotal: 0, items: [] } as never;
    }
    return nextOrder as never;
  });
});

afterEach(() => {
  cleanup();
  client.clear();
  usePendingPaymentStore.getState().clear();
  useCartStore.setState({ items: [] });
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
  vi.useRealTimers();
});

async function openPayment() {
  const hook = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {
    hook.result.current.setEmail(live.email);
    hook.result.current.setStep("payment");
  });
  return hook;
}

it("creates a guest QR payment and persists the receipt context", async () => {
  const { result } = await openPayment();
  await act(async () => result.current.handlePay());

  expect(result.current.step).toBe("qr");
  expect(usePendingPaymentStore.getState().pending).toEqual(live);
  expect(useCartStore.getState().items).toHaveLength(1);
  expect(api.post).toHaveBeenCalledWith("/orders", expect.objectContaining({
    recipientEmail: live.email, paymentMethod: "qrph",
    items: [{ photoId: "photo-1", eventId: "event-1" }],
  }), { headers: { "Idempotency-Key": expect.any(String) } });
});

it("restores a pending guest payment with its original amount and recipient", async () => {
  usePendingPaymentStore.getState().set(live);
  useCartStore.setState({ items: [] });
  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});

  expect(result.current.step).toBe("qr");
  expect(result.current.qrTotal).toBe(125);
  expect(result.current.qrEmail).toBe(live.email);
  expect(api.get).toHaveBeenCalledWith("/orders/order-1/status?token=guest-token");
});

it("snapshots a settled receipt before clearing the cart and pending payment", async () => {
  usePendingPaymentStore.getState().set(live);
  const paidAt = "2026-09-12T04:00:00Z";
  vi.mocked(api.get).mockResolvedValue({ id: live.orderId, status: "PAID", paidAt });
  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});

  expect(result.current.step).toBe("success");
  expect(result.current.receipt).toEqual({
    orderId: live.orderId, returnToken: live.returnToken, email: live.email,
    total: 125, itemCount: 1, paidAt,
  });
  expect(useCartStore.getState().items).toEqual([]);
  expect(usePendingPaymentStore.getState().pending).toBeNull();
});

it("lands directly on success for an already settled order without a QR", async () => {
  nextOrder = { ...nextOrder, status: "PAID", totalAmount: 0, checkoutTotalAmount: 0, qrPh: null, returnToken: "free-token" };
  const { result } = await openPayment();
  await act(async () => result.current.handlePay());

  expect(result.current.step).toBe("success");
  expect(result.current.receipt).toMatchObject({ total: 0, itemCount: 1, returnToken: "free-token" });
});

it("uses the confirmed order amount for the QR and subsequent receipt", async () => {
  nextOrder = { ...nextOrder, totalAmount: 90, checkoutTotalAmount: 375 };
  const { result } = await openPayment();
  expect(result.current.payable).toBe(125);
  await act(async () => result.current.handlePay());
  expect(result.current.qrTotal).toBe(375);
  expect(usePendingPaymentStore.getState().pending?.total).toBe(375);

  vi.mocked(api.get).mockResolvedValue({ id: live.orderId, status: "PAID", paidAt: null });
  await act(async () => result.current.handlePaidClaim());
  expect(result.current.receipt?.total).toBe(375);
});

it.each(["create", "status", "cancel"] as const)(
  "ignores a late %s failure after switching sessions",
  async (operation) => {
    if (operation !== "create") usePendingPaymentStore.getState().set(live);
    const { result } = await openPayment();
    if (operation !== "create") {
      await act(async () => result.current.setStep("qr"));
    }
    let reject!: (error: Error) => void;
    const response = new Promise<never>((_, fail) => { reject = fail; });
    if (operation === "status") vi.mocked(api.get).mockReturnValueOnce(response);
    else vi.mocked(api.post).mockReturnValueOnce(response);
    let action: Promise<void> | undefined;
    act(() => {
      if (operation === "create") action = result.current.handlePay();
      else if (operation === "cancel") action = result.current.handleCancelPayment();
      else result.current.handlePaidClaim();
    });
    const replacement = { ...live, orderId: "order-2", email: "other@example.test" };
    act(() => {
      useAuthStore.getState().setUser({
        id: "runner-2", email: replacement.email, name: "Other", role: "RUNNER",
        createdAt: "2026-09-12T00:00:00Z",
      });
      usePendingPaymentStore.getState().set(replacement);
    });
    await act(async () => {
      reject(new ApiError([{ code: "NOT_FOUND", message: "Old payment expired" }], 404));
      await action;
    });

    expect(usePendingPaymentStore.getState().pending).toEqual(replacement);
    expect(result.current.step).toBe("qr");
    expect(result.current.checkUnreachable).toBe(false);
    expect(result.current.qrCheckError).toBeNull();
    expect(result.current.outcome).toBeNull();
  },
);

it("retains the cart and idempotency key when creation fails and is retried", async () => {
  const { result } = await openPayment();
  vi.mocked(api.post).mockRejectedValueOnce(new Error("Offline"));
  await act(async () => result.current.handlePay());
  expect(result.current.step).toBe("payment");
  expect(result.current.outcome?.kind).toBe("error");
  expect(useCartStore.getState().items).toHaveLength(1);
  await act(async () => result.current.handlePay());

  const orders = vi.mocked(api.post).mock.calls.filter(([path]) => path === "/orders");
  expect(orders).toHaveLength(2);
  expect(orders[1]).toEqual(orders[0]);
  expect(result.current.step).toBe("qr");
});

function sentKeys(): string[] {
  return vi.mocked(api.post).mock.calls
    .filter(([path]) => path === "/orders")
    .map((call) => (call[2] as { headers: Record<string, string> }).headers["Idempotency-Key"]);
}

it("a reload with the same cart and email reuses the stored key", async () => {
  const first = await openPayment();
  vi.mocked(api.post).mockRejectedValueOnce(new Error("Offline"));
  await act(async () => first.result.current.handlePay());
  first.unmount();

  const second = await openPayment();
  await act(async () => second.result.current.handlePay());

  const [before, after] = sentKeys();
  expect(after).toBe(before);
});

it("changing the coupon or the email mints a new key", async () => {
  const { result } = await openPayment();
  vi.mocked(api.post).mockRejectedValue(new Error("Offline"));
  await act(async () => result.current.handlePay());
  await act(async () => result.current.setEmail("other@example.test"));
  await act(async () => result.current.handlePay());
  vi.mocked(api.post).mockResolvedValueOnce({ code: "SAVE10", discountTotal: 10, items: [] } as never);
  await act(async () => result.current.setCouponInput("save10"));
  await act(async () => result.current.handleApplyCoupon());
  await act(async () => result.current.handlePay());

  const [first, emailChanged, couponApplied] = sentKeys();
  expect(emailChanged).not.toBe(first);
  expect(couponApplied).not.toBe(emailChanged);
});

it("a CHECKOUT_IN_PROGRESS answer shows the in-progress outcome and the next attempt uses a new key", async () => {
  const { result } = await openPayment();
  const message = "These photos are already in a checkout that is waiting for payment.";
  vi.mocked(api.post).mockRejectedValueOnce(
    new ApiError([{ code: "CHECKOUT_IN_PROGRESS", message }], 409),
  );
  await act(async () => result.current.handlePay());

  expect(result.current.outcome).toEqual({ kind: "in_progress", message });
  expect(result.current.step).toBe("payment");
  await act(async () => result.current.handlePay());
  const [refused, next] = sentKeys();
  expect(next).not.toBe(refused);
});

it("a CONFLICT answer retires the key; a PAYMENT_FAILED or network failure keeps it", async () => {
  const { result } = await openPayment();
  vi.mocked(api.post)
    .mockRejectedValueOnce(new Error("Offline"))
    .mockRejectedValueOnce(new ApiError([{ code: "PAYMENT_FAILED", message: "Gateway down" }], 409))
    .mockRejectedValueOnce(new ApiError([{ code: "CONFLICT", message: "Key reused" }], 409));
  for (let i = 0; i < 4; i++) await act(async () => result.current.handlePay());

  const [offline, paymentFailed, conflict, next] = sentKeys();
  expect(paymentFailed).toBe(offline);
  expect(conflict).toBe(paymentFailed);
  expect(next).not.toBe(conflict);
});

it("success clears the stored key", async () => {
  nextOrder = { ...nextOrder, status: "PAID", qrPh: null };
  const first = await openPayment();
  await act(async () => first.result.current.handlePay());
  expect(first.result.current.step).toBe("success");
  first.unmount();

  useCartStore.setState({ items: [
    { photoId: "photo-1", eventId: "event-1", price: 125, thumbnailUrl: "" },
  ] });
  const second = await openPayment();
  await act(async () => second.result.current.handlePay());

  const [paid, next] = sentKeys();
  expect(next).not.toBe(paid);
});

it("a QR payment that settles clears the stored key", async () => {
  const first = await openPayment();
  await act(async () => first.result.current.handlePay());
  vi.mocked(api.get).mockResolvedValue({ id: live.orderId, status: "PAID", paidAt: null });
  await act(async () => first.result.current.handlePaidClaim());
  expect(first.result.current.step).toBe("success");
  first.unmount();

  useCartStore.setState({ items: [
    { photoId: "photo-1", eventId: "event-1", price: 125, thumbnailUrl: "" },
  ] });
  vi.mocked(api.get).mockResolvedValue({ id: live.orderId, status: "PENDING", paidAt: null });
  const second = await openPayment();
  await act(async () => second.result.current.handlePay());

  const [paid, next] = sentKeys();
  expect(next).not.toBe(paid);
});

it("a refused attempt with another cart shape keeps the key that can resume the first order", async () => {
  vi.mocked(api.post).mockImplementation(async (path) =>
    (path === "/coupons/preview" ? { code: "SAVE10", discountTotal: 10, items: [] } : nextOrder) as never);
  const first = await openPayment();
  await act(async () => first.result.current.setCouponInput("save10"));
  await act(async () => first.result.current.handleApplyCoupon());
  vi.mocked(api.post).mockRejectedValueOnce(new Error("Offline"));
  await act(async () => first.result.current.handlePay());
  first.unmount();

  vi.mocked(api.post).mockImplementation(async (path) =>
    (path === "/coupons/preview" ? { code: null, discountTotal: 0, items: [] } : nextOrder) as never);
  const second = await openPayment();
  vi.mocked(api.post).mockRejectedValueOnce(
    new ApiError([{ code: "CHECKOUT_IN_PROGRESS", message: "Already open" }], 409),
  );
  await act(async () => second.result.current.handlePay());
  vi.mocked(api.post).mockResolvedValueOnce({ code: "SAVE10", discountTotal: 10, items: [] } as never);
  await act(async () => second.result.current.setCouponInput("save10"));
  await act(async () => second.result.current.handleApplyCoupon());
  await act(async () => second.result.current.handlePay());

  const [withCoupon, withoutCoupon, couponAgain] = sentKeys();
  expect(withoutCoupon).not.toBe(withCoupon);
  expect(couponAgain).toBe(withCoupon);
});

it("checkout still works when sessionStorage throws, and a retry resends its key", async () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("blocked"); });
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("blocked"); });
  const { result } = await openPayment();
  vi.mocked(api.post).mockRejectedValueOnce(new Error("Offline"));
  await act(async () => result.current.handlePay());
  await act(async () => result.current.handlePay());

  expect(result.current.step).toBe("qr");
  const [failed, retry] = sentKeys();
  expect(retry).toBe(failed);
});

it.each(["PAID", "EXPIRED"] as const)("uses the backend's %s outcome after cancellation", async (status) => {
  usePendingPaymentStore.getState().set(live);
  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});
  vi.mocked(api.post).mockResolvedValueOnce({ id: live.orderId, status, paidAt: null });
  await act(async () => result.current.handleCancelPayment());

  expect(api.post).toHaveBeenCalledWith("/orders/order-1/cancel?token=guest-token");
  expect(result.current.step).toBe(status === "PAID" ? "success" : "payment");
  expect(useCartStore.getState().items).toHaveLength(status === "PAID" ? 0 : 1);
  expect(usePendingPaymentStore.getState().pending).toBeNull();
  if (status === "EXPIRED") expect(result.current.outcome).toEqual({ kind: "cancelled" });
});

it("keeps the QR and cart retryable after a cancellation failure", async () => {
  usePendingPaymentStore.getState().set(live);
  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});
  vi.mocked(api.post).mockRejectedValueOnce(new Error("Offline"));
  await act(async () => result.current.handleCancelPayment());

  expect(result.current.step).toBe("qr");
  expect(result.current.cancelling).toBe(false);
  expect(result.current.qrCheckError).toContain("couldn't cancel");
  expect(usePendingPaymentStore.getState().pending).toEqual(live);
  expect(useCartStore.getState().items).toHaveLength(1);
});

it("stops polling when the guest capability expires without clearing the cart", async () => {
  usePendingPaymentStore.getState().set(live);
  vi.mocked(api.get).mockRejectedValue(new ApiError([{ code: "NOT_FOUND", message: "Expired" }], 404));
  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});

  expect(result.current.checkUnreachable).toBe(true);
  expect(usePendingPaymentStore.getState().pending).toBeNull();
  expect(useCartStore.getState().items).toHaveLength(1);
});

it("verifies a paid claim and removes polling timers on unmount", async () => {
  vi.useFakeTimers();
  usePendingPaymentStore.getState().set(live);
  const { result, unmount } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  await act(async () => {});
  await act(async () => result.current.handlePaidClaim());
  expect(api.get).toHaveBeenCalledWith("/orders/order-1/status?token=guest-token&verify=true");
  expect(usePendingPaymentStore.getState().pending?.paidClaimedAt).not.toBeNull();
  await act(async () => vi.advanceTimersByTimeAsync(2500));
  const requests = vi.mocked(api.get).mock.calls.length;
  unmount();
  await act(async () => vi.advanceTimersByTimeAsync(30_000));
  expect(api.get).toHaveBeenCalledTimes(requests);
});
