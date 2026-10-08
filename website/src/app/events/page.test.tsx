import { afterEach, expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { api } from "@/lib/api";
import EventsPage from "./page";
import type { ListEvent } from "@/types/event";

function seed(page: ReactElement<{ children: ReactElement[] }>): ListEvent[] | undefined {
  const suspense = page.props.children[1] as ReactElement<{ children: ReactElement<{ events?: ListEvent[] }> }>;
  return suspense.props.children.props.events;
}
afterEach(() => vi.restoreAllMocks());
it("seeds SSR with the complete catalog including later pages", async () => {
  const row = { id: "a", slug: "a", name: "Race", date: "2099-01-01", location: "Venue", status: "ACTIVE", photoCount: 0, participantCount: 0 };
  vi.spyOn(api, "get").mockResolvedValueOnce({ items: [row], total: 2, offset: 0, limit: 200 }).mockResolvedValueOnce({ items: [{ ...row, id: "b" }], total: 2, offset: 1, limit: 200 });
  expect(seed(await EventsPage())?.map(e => e.id)).toEqual(["a", "b"]);
});
it("leaves SSR unseeded on failure so the client can show loading/error/retry", async () => {
  vi.spyOn(api, "get").mockRejectedValue(new Error("Unavailable"));
  expect(seed(await EventsPage())).toBeUndefined();
});
