import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { fetchPhotoMetadata } from "@/lib/api-photos";
import { useAuthStore } from "@/store/auth-store";
import { usePhotoMetadata } from "./use-photo-metadata";

let client: QueryClient;
function wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
function login(id: string, role: "PHOTOGRAPHER" | "RUNNER" = "PHOTOGRAPHER") {
  useAuthStore.getState().startSession({ id, role, name: id, email: `${id}@example.test`, createdAt: "2026-09-10" });
}
beforeEach(() => {
  useAuthStore.getState().logout();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("requests the exact encoded metadata route for each audience and forwards cancellation", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ photoId: "photo / one", status: "ready", fields: null });
  const signal = new AbortController();
  await fetchPhotoMetadata("photo / one", "owner", signal.signal);
  await fetchPhotoMetadata("photo / one", "public", signal.signal);
  expect(get.mock.calls).toEqual([
    ["/me/photographer/photos/photo%20%2F%20one/metadata", { signal: signal.signal }],
    ["/public/photos/photo%20%2F%20one/metadata", { signal: signal.signal }],
  ]);
});

it("waits for an open dialog and a known access route; permits public guests", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ photoId: "a", status: "empty", fields: null });
  const initialProps: { access: "none" | "public" | "owner" | undefined; isOpen: boolean } = { access: undefined, isOpen: false };
  const hook = renderHook(({ access, isOpen }: { access: "none" | "public" | "owner" | undefined; isOpen: boolean }) =>
    usePhotoMetadata("a", access, isOpen), { initialProps, wrapper });
  hook.rerender({ access: "none", isOpen: true });
  hook.rerender({ access: "public", isOpen: false });
  expect(get).not.toHaveBeenCalled();
  hook.rerender({ access: "public", isOpen: true });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  expect(get.mock.calls[0][0]).toBe("/public/photos/a/metadata");
  expect(client.getQueryCache().getAll()[0].queryKey.slice(0, 4)).toEqual(["photo", "metadata", "public", "a"]);
  expect(client.getQueryCache().getAll()[0].options).toMatchObject({ staleTime: 0, gcTime: 0, refetchOnMount: "always" });
  act(() => login("runner-a", "RUNNER"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(get.mock.calls[1][0]).toBe("/public/photos/a/metadata");
  expect(client.getQueryCache().getAll().some((q) => q.queryKey.includes("runner-a"))).toBe(true);
});

it("routes owner reads only for photographers and keeps photo and session results separate", async () => {
  const get = vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve({ photoId: path, status: "ready", fields: null }));
  login("runner", "RUNNER");
  const hook = renderHook(({ photoId }: { photoId: string }) => usePhotoMetadata(photoId, "owner", true),
    { initialProps: { photoId: "a" }, wrapper });
  expect(get).not.toHaveBeenCalled();
  act(() => login("photographer-a"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const firstKey = client.getQueryCache().getAll().find((q) => q.queryKey[3] === "a" && q.state.data)?.queryKey;
  hook.rerender({ photoId: "b" });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(hook.result.current.data?.photoId).toContain("/b/metadata");
  const previousSessionKey = client.getQueryCache().getAll().find((q) => q.queryKey.includes("photographer-a") && q.queryKey[3] === "b")?.queryKey;
  act(() => login("photographer-b"));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(3));
  const secondKey = client.getQueryCache().getAll().find((q) => q.queryKey.includes("photographer-b") && q.queryKey[3] === "b")?.queryKey;
  expect(secondKey).toBeDefined();
  expect(secondKey).not.toEqual(previousSessionKey);
  expect(secondKey).not.toEqual(firstKey);
  expect(client.getQueryCache().getAll().some((q) => q.queryKey[3] === "a" && q.state.data)).toBe(false);
});

it("aborts a metadata request when the dialog closes", async () => {
  const get = vi.spyOn(api, "get").mockImplementation(() => new Promise(() => {}));
  const hook = renderHook(({ isOpen }) => usePhotoMetadata("a", "public", isOpen),
    { initialProps: { isOpen: true }, wrapper });
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const signal = get.mock.calls[0][1]?.signal;
  expect(signal).toBeInstanceOf(AbortSignal);
  hook.rerender({ isOpen: false });
  await waitFor(() => expect(signal?.aborted).toBe(true));
});
