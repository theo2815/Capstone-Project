import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { Order } from "@/types/order";
import type { OrderStatusPayload } from "@/lib/api-orders";
import { checkoutKeys, useCheckoutActions } from "./use-checkout-actions";

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => {
    resolve = yes;
  });
  return { promise, resolve };
}

let client: QueryClient;

afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
});

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  useAuthStore.getState().setUser({
    id: "runner-1",
    email: "runner@example.com",
    name: "Runner",
    role: "RUNNER",
    createdAt: "2026-09-01T00:00:00Z",
  });
});

const order: Order = {
  id: "order-1",
  status: "PENDING",
  items: [],
  totalAmount: 125,
  paymentMethod: "qrph",
  createdAt: "2026-09-01T00:00:00Z",
  qrPh: {
    imageUrl: "data:image/png;base64,cXJwaA==",
    expiresAt: "2026-09-01T01:00:00Z",
    returnToken: "return-token",
    testUrl: null,
  },
};

it("invalidates canonical order cache only after backend-confirmed order creation", async () => {
  client.setQueryData(checkoutKeys.orders, [{ id: "cached-order" }]);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  vi.spyOn(api, "post").mockResolvedValue(order);
  const { result } = renderHook(useCheckoutActions, { wrapper: Wrapper });
  let created = null as Order | null;

  await act(async () => {
    created = await result.current.createOrder({
      items: [{ photoId: "photo-1", eventId: "event-1" }],
      paymentMethod: "qrph",
      idempotencyKey: "id",
    });
  });

  expect(created).toEqual(order);
  expect(invalidate).toHaveBeenCalledWith({ queryKey: checkoutKeys.orders });
});

it("returns rejected promises on backend failure and keeps canonical query data intact", async () => {
  const baseline = { id: "cached-order" };
  client.setQueryData(checkoutKeys.orders, baseline);
  vi.spyOn(api, "post").mockRejectedValue(new Error("Backend unavailable"));
  const { result } = renderHook(useCheckoutActions, { wrapper: Wrapper });

  await expect(
    result.current.createOrder({ items: [], paymentMethod: "qrph", idempotencyKey: "id" }),
  ).rejects.toThrow("Backend unavailable");

  expect(client.getQueryData(checkoutKeys.orders)).toEqual(baseline);
});

it.each(["checkStatus", "cancelPayment"] as const)("discards stale %s results after auth transition", async (method) => {
  const status: OrderStatusPayload = { id: "order-1", status: "PAID", paidAt: "2026-09-01T00:00:00Z" };
  const response = deferred<typeof status>();

  const spy = method === "checkStatus"
    ? vi.spyOn(api, "get").mockImplementation(async () => response.promise)
    : vi.spyOn(api, "post").mockImplementation(async () => response.promise);

  const { result } = renderHook(useCheckoutActions, { wrapper: Wrapper });
  let operation!: Promise<OrderStatusPayload | null>;
  act(() => {
    if (method === "checkStatus") {
      operation = result.current.checkStatus({ orderId: "order-1", returnToken: "return-token" });
    } else {
      operation = result.current.cancelPayment({ orderId: "order-1", returnToken: "return-token" });
    }
    useAuthStore.getState().logout();
    useAuthStore.getState().setUser({
      id: "runner-2",
      email: "other@example.com",
      name: "Other",
      role: "RUNNER",
      createdAt: "2026-09-01T00:00:00Z",
    });
  });

  await act(async () => {
    response.resolve(status);
    expect(await operation).toBeNull();
  });

  expect(spy).toHaveBeenCalledTimes(1);
  expect(client.getQueryCache().getAll()).toHaveLength(0);
});

it("discards a createOrder confirmation after auth transition", async () => {
  const response = deferred<Order>();
  vi.spyOn(api, "post").mockImplementation(async () => response.promise);
  const { result } = renderHook(useCheckoutActions, { wrapper: Wrapper });
  let operation!: Promise<Order | null>;
  act(() => {
    operation = result.current.createOrder({
      items: [{ photoId: "photo-1", eventId: "event-1" }],
      paymentMethod: "qrph",
      idempotencyKey: "id",
    });
    useAuthStore.getState().logout();
    useAuthStore.getState().setUser({
      id: "runner-2",
      email: "other@example.com",
      name: "Other",
      role: "RUNNER",
      createdAt: "2026-09-01T00:00:00Z",
    });
  });

  await act(async () => {
    response.resolve(order);
    expect(await operation).toBeNull();
  });
});
