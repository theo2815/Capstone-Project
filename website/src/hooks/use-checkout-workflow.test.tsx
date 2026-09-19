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
  useAuthStore.getState().logout();
  useCartStore.setState({ items: [
    { photoId: "photo-1", eventId: "event-1", price: 125, thumbnailUrl: "" },
  ] });
  live = {
    orderId: "order-1", imageUrl: "data:image/png;base64,cXI=",
    expiresAt: new Date(Date.now() + 600_000).toISOString(),
    returnToken: "guest-token", testUrl: null, email: "runner@example.test",
    total: 125, itemCount: 1, paidClaimedAt: null,
  };
  usePendingPaymentStore.getState().clear();
  nextOrder = {
    id: live.orderId, status: "PENDING", items: [], totalAmount: 125,
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
  nextOrder = { ...nextOrder, status: "PAID", totalAmount: 0, qrPh: null, returnToken: "free-token" };
  const { result } = await openPayment();
  await act(async () => result.current.handlePay());

  expect(result.current.step).toBe("success");
  expect(result.current.receipt).toMatchObject({ total: 0, itemCount: 1, returnToken: "free-token" });
});

it("uses the confirmed order amount for the QR and subsequent receipt", async () => {
  nextOrder = { ...nextOrder, totalAmount: 90 };
  const { result } = await openPayment();
  expect(result.current.payable).toBe(125);
  await act(async () => result.current.handlePay());
  expect(result.current.qrTotal).toBe(90);
  expect(usePendingPaymentStore.getState().pending?.total).toBe(90);

  vi.mocked(api.get).mockResolvedValue({ id: live.orderId, status: "PAID", paidAt: null });
  await act(async () => result.current.handlePaidClaim());
  expect(result.current.receipt?.total).toBe(90);
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
