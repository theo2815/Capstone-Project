import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import { useAdminMutation, type AdminTarget } from "./use-admin-mutation";

const admin = { id: "admin-a", role: "ADMIN" as const, name: "Admin", email: "admin@example.test", createdAt: "2026-10-08T00:00:00Z" };
const row: AdminTarget = { kind: "user", id: "one" };
const other: AdminTarget = { kind: "user", id: "two" };
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

describe("shared admin mutation ownership", () => {
  let client: QueryClient;
  function wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  }
  beforeEach(() => {
    useAuthStore.getState().startSession(admin);
    client = new QueryClient();
  });
  afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); });

  it("blocks_same_tick_and_cross_instance_submission", async () => {
    const response = deferred<string>();
    const execute = vi.fn(() => response.promise);
    const views: Record<string, ReturnType<typeof useAdminMutation>> = {};
    function View({ name }: { name: string }) {
      const guard = useAdminMutation(); views[name] = guard;
      return <output aria-label={name}>{guard.pendingAction([row]) ?? "idle"}</output>;
    }
    render(<><View name="first" /><View name="second" /></>, { wrapper });
    let accepted!: ReturnType<typeof views.first.run<string>>;
    let blocked!: ReturnType<typeof views.first.run<string>>;
    act(() => {
      accepted = views.first.run({ targets: [row], action: "approve", execute });
      expect(views.second.isPending([row])).toBe(true);
      blocked = views.second.run({ targets: [row], action: "approve", execute });
    });
    await expect(blocked).resolves.toEqual({ kind: "blocked" });
    await waitFor(() => {
      expect(screen.getByLabelText("first")).toHaveTextContent("approve");
      expect(screen.getByLabelText("second")).toHaveTextContent("approve");
    });
    expect(execute).toHaveBeenCalledTimes(1);
    await act(async () => { response.resolve("saved"); expect(await accepted).toEqual({ kind: "completed", value: "saved" }); });
    await waitFor(() => {
      expect(screen.getByLabelText("first")).toHaveTextContent("idle");
      expect(screen.getByLabelText("second")).toHaveTextContent("idle");
    });
  });

  it("blocks_conflicting_verb_and_overlapping_bulk", async () => {
    const response = deferred<number>();
    const execute = vi.fn(() => response.promise);
    const refused = vi.fn(async () => 99);
    const { result } = renderHook(() => ({ row: useAdminMutation(), batch: useAdminMutation() }), { wrapper });
    let accepted!: Promise<unknown>;
    act(() => { accepted = result.current.row.run({ targets: [row], action: "approve", execute }); });
    await act(async () => {
      expect(await result.current.batch.run({ targets: [row], action: "reject", execute: refused })).toEqual({ kind: "blocked" });
      expect(await result.current.batch.run({ targets: [other, row], action: "suspend", execute: refused })).toEqual({ kind: "blocked" });
      // Same ID in another resource family is independent.
      expect(await result.current.batch.run({ targets: [{ kind: "event", id: "one" }], action: "edit", execute: async () => 3 })).toEqual({ kind: "completed", value: 3 });
      expect(await result.current.batch.run({ targets: [other], action: "reject", execute: async () => 4 })).toEqual({ kind: "completed", value: 4 });
    });
    expect(refused).not.toHaveBeenCalled();
    await act(async () => { response.resolve(1); await accepted; });
  });

  it("reserves every bulk target before executing its callback", async () => {
    const response = deferred<void>();
    const refused = vi.fn(async () => undefined);
    const { result } = renderHook(() => ({ first: useAdminMutation(), second: useAdminMutation() }), { wrapper });
    const claims: boolean[] = [];
    const execute = vi.fn(() => {
      claims.push(result.current.second.isPending([row]), result.current.second.isPending([other]));
      return response.promise;
    });
    let accepted!: Promise<unknown>;
    act(() => { accepted = result.current.first.run({ targets: [row, other], action: "hold", execute }); });
    await act(async () => {
      expect(await result.current.second.run({ targets: [other], action: "release", execute: refused })).toEqual({ kind: "blocked" });
    });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(claims).toEqual([true, true]);
    expect(refused).not.toHaveBeenCalled();
    await act(async () => { response.resolve(); await accepted; });
  });

  it("releases rejected and synchronously thrown commands without retrying", async () => {
    const response = deferred<void>();
    const execute = vi.fn(() => response.promise);
    const { result } = renderHook(useAdminMutation, { wrapper });
    let accepted!: Promise<unknown>;
    act(() => { accepted = result.current.run({ targets: [row], action: "approve", execute }); });
    await waitFor(() => expect(result.current.isPending([row])).toBe(true));
    const error = new Error("Refused");
    const rejected = expect(accepted).rejects.toBe(error);
    await act(async () => { response.reject(error); await rejected; });
    expect(execute).toHaveBeenCalledTimes(1);
    expect(result.current.isPending([row])).toBe(false);
    await act(async () => {
      await expect(result.current.run({ targets: [row], action: "approve", execute: () => { throw error; } })).rejects.toBe(error);
      expect(await result.current.run({ targets: [row], action: "approve", execute: async () => "retry" })).toEqual({ kind: "completed", value: "retry" });
    });
  });

  it("keeps claims across unmount and remount until the command settles", async () => {
    const response = deferred<void>();
    const first = renderHook(useAdminMutation, { wrapper });
    let accepted!: Promise<unknown>;
    act(() => { accepted = first.result.current.run({ targets: [row], action: "delete", execute: () => response.promise }); });
    first.unmount();
    const second = renderHook(useAdminMutation, { wrapper });
    expect(second.result.current.pendingAction([row])).toBe("delete");
    const refused = vi.fn(async () => undefined);
    await expect(second.result.current.run({ targets: [row], action: "edit", execute: refused })).resolves.toEqual({ kind: "blocked" });
    expect(refused).not.toHaveBeenCalled();
    await act(async () => { response.resolve(); await accepted; });
    await waitFor(() => expect(second.result.current.isPending([row])).toBe(false));
  });

  it.each(["logout", "other-admin", "same-admin", "role-change"] as const)("isolates claims and stale callbacks after %s", async (change) => {
    const response = deferred<string>();
    const { result } = renderHook(useAdminMutation, { wrapper });
    const oldRun = result.current.run;
    let accepted!: Promise<unknown>;
    act(() => { accepted = oldRun({ targets: [row], action: "approve", execute: () => response.promise }); });
    await waitFor(() => expect(result.current.isPending([row])).toBe(true));
    act(() => {
      if (change === "logout") useAuthStore.getState().logout();
      else useAuthStore.getState().startSession({ ...admin, id: change === "other-admin" ? "admin-b" : admin.id, role: change === "role-change" ? "RUNNER" : "ADMIN" });
    });
    expect(result.current.isPending([row])).toBe(false);
    expect(result.current.pendingAction([row])).toBeNull();
    const refused = vi.fn(async () => "bad");
    await expect(oldRun({ targets: [row], action: "reject", execute: refused })).resolves.toEqual({ kind: "stale" });
    expect(refused).not.toHaveBeenCalled();
    await act(async () => {
      expect(await result.current.run({ targets: [row], action: "reject", execute: async () => "new" })).toEqual(change === "logout" || change === "role-change" ? { kind: "stale" } : { kind: "completed", value: "new" });
      response.resolve("old");
      expect(await accepted).toEqual({ kind: "stale" });
    });
  });

  it("does not invoke a callback after a same-tick session change, and suppresses stale rejection", async () => {
    const response = deferred<void>();
    const execute = vi.fn(() => response.promise);
    const { result } = renderHook(useAdminMutation, { wrapper });
    let accepted!: Promise<unknown>;
    act(() => {
      accepted = result.current.run({ targets: [row], action: "delete", execute });
      useAuthStore.getState().startSession(admin);
    });
    await act(async () => { response.resolve(); expect(await accepted).toEqual({ kind: "stale" }); });
    expect(execute).not.toHaveBeenCalled();
    const rejectedResponse = deferred<void>();
    execute.mockImplementation(() => rejectedResponse.promise);
    act(() => { accepted = result.current.run({ targets: [row], action: "delete", execute }); });
    await waitFor(() => expect(execute).toHaveBeenCalledTimes(1));
    act(() => { useAuthStore.getState().logout(); client.clear(); });
    await act(async () => { rejectedResponse.reject(new Error("Old failure")); expect(await accepted).toEqual({ kind: "stale" }); });
    expect(result.current.isPending([row])).toBe(false);
  });

  it.each(["success", "failure"])("returns stale for old-session %s without releasing a new claim", async (outcome) => {
    const oldResponse = deferred<string>();
    const newResponse = deferred<string>();
    const { result } = renderHook(useAdminMutation, { wrapper });
    let oldRequest!: Promise<unknown>;
    let newRequest!: Promise<unknown>;
    act(() => { oldRequest = result.current.run({ targets: [row], action: "approve", execute: () => oldResponse.promise }); });
    await waitFor(() => expect(result.current.pendingAction([row])).toBe("approve"));
    act(() => useAuthStore.getState().startSession(admin));
    act(() => { newRequest = result.current.run({ targets: [row], action: "reject", execute: () => newResponse.promise }); });
    const stale = expect(oldRequest).resolves.toEqual({ kind: "stale" });
    await act(async () => {
      if (outcome === "success") oldResponse.resolve("old");
      else oldResponse.reject(new Error("Old failure"));
      await stale;
    });
    expect(result.current.pendingAction([row])).toBe("reject");
    await act(async () => { newResponse.resolve("new"); expect(await newRequest).toEqual({ kind: "completed", value: "new" }); });
  });
});
