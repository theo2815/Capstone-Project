import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { PhotographerPayout } from "@/lib/photographer-mock";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import {
  photographerPayoutKeys,
  usePhotographerPayoutActions,
} from "./use-photographer-payout-actions";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
}

const photographer = {
  id: "photographer-1",
  role: "PHOTOGRAPHER" as const,
  name: "Maya Runner",
  email: "maya@example.test",
  createdAt: "2026-09-10T00:00:00Z",
};

const openRequest: PhotographerPayout = {
  id: "cycle-1",
  weekOf: "2026-09-07",
  requestedAt: "2026-09-10T00:00:00Z",
  amount: 1500,
  status: "pending",
  settledAt: null,
  method: "gcash",
  reference: null,
  holdReason: null,
};

const baseBalance = {
  unpaidBalance: 2500,
  clearing: 0,
  reserved: 1000,
  accountCooldownUntil: null,
  minimum: 500,
  hasOpenRequest: true,
  openRequest,
  primaryAccount: {
    method: "gcash" as const,
    accountNumber: "0917-000-1111",
    accountName: "Test User",
    hasQr: true,
  },
};

let client: QueryClient;
function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function currentKeys() {
  const session = useAuthStore.getState();
  if (!session.user) throw new Error("No session user");
  if (session.user.role !== "PHOTOGRAPHER") throw new Error("Not a photographer session");

  return {
    list: photographerPayoutKeys.list(session.user.id, session.user.role, session.sessionVersion),
    balance: photographerPayoutKeys.balance(session.user.id, session.user.role, session.sessionVersion),
  };
}

beforeEach(() => {
  useAuthStore.getState().setUser(photographer);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(() => {
  cleanup();
  client.clear();
  useAuthStore.getState().logout();
  useToastStore.getState().clear();
  vi.restoreAllMocks();
});

describe("request payout", () => {
  it("keeps canonical cache untouched and shows error toast on backend failure", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Backend unavailable"));

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    const success = await act(async () => result.current.request());

    expect(success).toBe(false);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(api.post).toHaveBeenCalledWith("/me/photographer/payouts/request");
    expect(client.getQueryData(list)).toEqual([openRequest]);
    expect(client.getQueryData(balance)).toEqual(baseBalance);
    expect(invalidate).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([
      expect.objectContaining({
        kind: "error",
        message: "Couldn't request payout — please try again.",
      }),
    ]);
  });

  it("returns true and invalidates payout and balance queries after confirmation", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    vi.spyOn(api, "post").mockResolvedValueOnce(openRequest);

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    const success = await act(async () => result.current.request());

    expect(success).toBe(true);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(api.post).toHaveBeenCalledWith("/me/photographer/payouts/request");
    expect(invalidate).toHaveBeenCalledWith({ queryKey: list });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: balance });
    expect(client.getQueryData(list)).toEqual([openRequest]);
    expect(client.getQueryData(balance)).toEqual(baseBalance);
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it("discards backend confirmation after session change and does not mutate cache", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const response = deferred<PhotographerPayout>();
    vi.spyOn(api, "post").mockReturnValueOnce(response.promise);

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    let pending!: Promise<boolean>;
    act(() => {
      pending = result.current.request();
    });

    act(() => {
      useAuthStore.getState().logout();
      client.clear();
      useAuthStore.getState().setUser(photographer);
    });
    const next = currentKeys();
    client.setQueryData(next.list, [{ ...openRequest, id: "cycle-next" }]);
    const nextState = client.getQueryData(next.list);

    await act(async () => {
      response.resolve(openRequest);
      expect(await pending).toBe(false);
    });

    expect(client.getQueryData(next.list)).toEqual(nextState);
    expect(invalidate).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([]);
  });
});

describe("withdraw payout", () => {
  it("keeps canonical cache untouched and shows error toast on backend failure", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Backend unavailable"));

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    const success = await act(async () => result.current.withdraw("cycle-1"));

    expect(success).toBe(false);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(api.post).toHaveBeenCalledWith("/me/photographer/payouts/cycle-1/withdraw");
    expect(client.getQueryData(list)).toEqual([openRequest]);
    expect(client.getQueryData(balance)).toEqual(baseBalance);
    expect(invalidate).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([
      expect.objectContaining({
        kind: "error",
        message: "Couldn't withdraw payout — please try again.",
      }),
    ]);
  });

  it("returns true and invalidates payout and balance queries after confirmation", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    vi.spyOn(api, "post").mockResolvedValueOnce(undefined);

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    const success = await act(async () => result.current.withdraw("cycle-1"));

    expect(success).toBe(true);
    expect(api.post).toHaveBeenCalledTimes(1);
    expect(api.post).toHaveBeenCalledWith("/me/photographer/payouts/cycle-1/withdraw");
    expect(invalidate).toHaveBeenCalledWith({ queryKey: list });
    expect(invalidate).toHaveBeenCalledWith({ queryKey: balance });
    expect(client.getQueryData(list)).toEqual([openRequest]);
    expect(client.getQueryData(balance)).toEqual(baseBalance);
    expect(useToastStore.getState().toasts).toEqual([]);
  });

  it("discards backend confirmation after session change and does not mutate cache", async () => {
    const { list, balance } = currentKeys();
    client.setQueryData(list, [openRequest]);
    client.setQueryData(balance, baseBalance);
    const invalidate = vi.spyOn(client, "invalidateQueries");
    const response = deferred<void>();
    vi.spyOn(api, "post").mockReturnValueOnce(response.promise);

    const { result } = renderHook(usePhotographerPayoutActions, { wrapper: Wrapper });
    let pending!: Promise<boolean>;
    act(() => {
      pending = result.current.withdraw("cycle-1");
    });

    act(() => {
      useAuthStore.getState().logout();
      client.clear();
      useAuthStore.getState().setUser(photographer);
    });
    const next = currentKeys();
    client.setQueryData(next.list, [{ ...openRequest, id: "cycle-next" }]);
    const nextState = client.getQueryData(next.list);

    await act(async () => {
      response.resolve();
      expect(await pending).toBe(false);
    });

    expect(client.getQueryData(next.list)).toEqual(nextState);
    expect(invalidate).not.toHaveBeenCalled();
    expect(useToastStore.getState().toasts).toEqual([]);
  });
});
