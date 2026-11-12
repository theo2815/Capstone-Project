import { act, cleanup, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { useEventLivePhotos } from "./use-event-live-photos";

vi.mock("@/lib/auth", () => ({ getAccessToken: () => "fixture-token" }));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const page = (ids: string[], total: number) => ({
  items: ids.map((id) => ({ id })), total, offset: 0, limit: 20,
});

function setup() {
  const socket = { onopen: () => {}, onmessage: vi.fn<(event: { data: string }) => void>(), onclose: () => {}, onerror: () => {}, close: vi.fn() };
  vi.stubGlobal("WebSocket", vi.fn(function () { return socket; }));
  const client = new QueryClient();
  const key = ["events", "race", "photos", { bib: null, snapshotAt: null }, "viewer"];
  client.setQueryData(key, { pages: [page(["a", "b"], 3), { ...page(["c"], 3), offset: 2 }], pageParams: [0, 2] });
  const fetcher = vi.spyOn(client, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  const hook = renderHook(() => useEventLivePhotos({ slug: "race", eventId: "ev-1", enabled: true }), { wrapper });
  const send = (message: unknown) => act(() => socket.onmessage({ data: JSON.stringify(message) }));
  const cached = () => client.getQueryData<{ pages: ReturnType<typeof page>[] }>(key)!;
  return { client, hook, send, cached, fetcher };
}

it("a photo.hidden frame removes the photo from the loaded pages without a refetch", () => {
  const { client, send, cached, fetcher, hook } = setup();
  send({ type: "photo.hidden", photoId: "b" });
  expect(cached().pages.map((p) => p.items.map((i) => i.id))).toEqual([["a"], ["c"]]);
  expect(cached().pages.map((p) => p.total)).toEqual([2, 2]);
  expect(fetcher).not.toHaveBeenCalled();
  expect(hook.result.current.newCount).toBe(0);
  client.clear();
});

it("a photo.hidden frame for a photo that is not loaded changes nothing", () => {
  const { client, send, cached } = setup();
  send({ type: "photo.hidden", photoId: "zzz" });
  expect(cached().pages.map((p) => p.total)).toEqual([3, 3]);
  client.clear();
});

it("an unknown frame type is still ignored", () => {
  const { client, send, cached, hook } = setup();
  send({ type: "photo.mystery", photoId: "a" });
  expect(cached().pages[0].items.map((i) => i.id)).toEqual(["a", "b"]);
  expect(hook.result.current.newCount).toBe(0);
  client.clear();
});
