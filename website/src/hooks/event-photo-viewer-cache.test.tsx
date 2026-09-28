import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { MockPhoto } from "@/types/photo";
import { useEventPhotos, usePublicPhotographerEventPhotos } from "./use-event-photos";

let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
function login(id: string) {
  useAuthStore.getState().startSession({ id, role: "RUNNER", name: id, email: `${id}@example.test`, createdAt: "2026-09-10" });
}
const photo = (id: string): MockPhoto => ({ id, bib: null, km: null, time: "10:00", tone: 0, span: "default", price: 0 });
const page = (id: string, offset = 0, snapshotAt?: string) => ({ items: [photo(id)], total: 2, offset, limit: 1, snapshotAt });
beforeEach(() => {
  useAuthStore.getState().logout();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("uses the anonymous page-zero seed only for guests and fetches authenticated hydration", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue(page("private", 0, "new-snapshot"));
  const hook = renderHook(() => useEventPhotos({ slug: "race", initialPage: page("public", 0, "anonymous-snapshot") }), { wrapper });
  expect(hook.result.current.photos.map((p) => p.id)).toEqual(["public"]);
  expect(get).not.toHaveBeenCalled();
  act(() => login("runner-a"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(hook.result.current.photos.map((p) => p.id)).toEqual(["private"]));
  expect(client.getQueryCache().getAll().map((q) => q.queryKey.at(-3))).toContain("runner-a");
});

it("resets the paging snapshot and cache on viewer changes and forwards abort signals", async () => {
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(page("a-zero", 0, "a-snapshot"))
    .mockResolvedValueOnce(page("a-one", 1, "a-snapshot"))
    .mockResolvedValueOnce(page("b-zero", 0, "b-snapshot"))
    .mockResolvedValueOnce(page("b-one", 1, "b-snapshot"));
  login("runner-a");
  const hook = renderHook(() => useEventPhotos({ slug: "race" }), { wrapper });
  await waitFor(() => expect(hook.result.current.photos.map((p) => p.id)).toEqual(["a-zero"]));
  act(() => hook.result.current.fetchNextPage());
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(get.mock.calls[1][0]).toContain("snapshotAt=a-snapshot");
  expect(get.mock.calls[0][1]?.signal).toBeInstanceOf(AbortSignal);
  act(() => login("runner-b"));
  await waitFor(() => expect(hook.result.current.photos.map((p) => p.id)).toEqual(["b-zero"]));
  act(() => hook.result.current.fetchNextPage());
  await waitFor(() => expect(get).toHaveBeenCalledTimes(4));
  expect(get.mock.calls[3][0]).toContain("snapshotAt=b-snapshot");
  expect(client.getQueryCache().getAll().map((q) => q.queryKey).some((key) => key.includes("runner-a"))).toBe(true);
  expect(client.getQueryCache().getAll().map((q) => q.queryKey).some((key) => key.includes("runner-b"))).toBe(true);
});

it("ignores a retired viewer's late snapshot when paging the new viewer", async () => {
  let finishOld!: (value: ReturnType<typeof page>) => void;
  const old = new Promise<ReturnType<typeof page>>((resolve) => { finishOld = resolve; });
  const get = vi.spyOn(api, "get")
    .mockReturnValueOnce(old)
    .mockResolvedValueOnce(page("b-zero", 0, "b-snapshot"))
    .mockResolvedValueOnce(page("b-one", 1, "b-snapshot"));
  login("runner-a");
  const hook = renderHook(() => useEventPhotos({ slug: "race" }), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  act(() => login("runner-b"));
  await waitFor(() => expect(hook.result.current.photos.map((p) => p.id)).toEqual(["b-zero"]));
  await act(async () => finishOld(page("a-zero", 0, "a-snapshot")));
  act(() => hook.result.current.fetchNextPage());
  await waitFor(() => expect(get).toHaveBeenCalledTimes(3));
  expect(get.mock.calls[2][0]).toContain("snapshotAt=b-snapshot");
});

it("scopes photographer gallery pages by viewer and cancels retired requests", async () => {
  const get = vi.spyOn(api, "get").mockImplementation(() => new Promise(() => {}));
  const hook = renderHook(() => usePublicPhotographerEventPhotos("handle", "race"), { wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const firstSignal = get.mock.calls[0][1]?.signal;
  expect(firstSignal).toBeInstanceOf(AbortSignal);
  act(() => login("runner-a"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(firstSignal?.aborted).toBe(true);
  expect(client.getQueryCache().getAll().map((q) => q.queryKey)).toEqual(expect.arrayContaining([
    expect.arrayContaining(["runner-a"]),
  ]));
  hook.unmount();
  expect(get.mock.calls[1][1]?.signal?.aborted).toBe(true);
});
