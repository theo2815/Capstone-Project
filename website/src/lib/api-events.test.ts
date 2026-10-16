import { afterEach, expect, it, vi } from "vitest";
import { api } from "./api";
import { fetchEventsList, fetchEventsPage, fetchPublicEventsCatalog } from "./api-events";
import type { Event } from "@/types/event";

const event = (id: string): Event => ({ id, slug: id, name: id, date: "2099-01-01", location: "Venue, Cebu", status: "ACTIVE", photoCount: 0, participantCount: 0, regionCode: "region-vii", provinceCode: "cebu" });
const page = (ids: string[], total: number, offset = 0) => ({ items: ids.map(event), total, offset, limit: 200 });
afterEach(() => vi.restoreAllMocks());

it("keeps the compatibility wrapper to one requested page and public status set", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue(page(["a"], 201, 20));
  expect(await fetchEventsList({ offset: 20, limit: 10, city: "Cebu", search: "race" })).toEqual([expect.objectContaining({ id: "a", city: "Cebu", state: "upcoming" })]);
  expect(get).toHaveBeenCalledTimes(1);
  const params = new URLSearchParams(get.mock.calls[0][0].split("?")[1]);
  expect(Object.fromEntries(params)).toEqual({ status: "ACTIVE,COMPLETED,ARCHIVED", offset: "20", limit: "10", city: "Cebu", search: "race" });
});
it("exposes page metadata and forwards cancellation", async () => {
  const signal = new AbortController().signal;
  const get = vi.spyOn(api, "get").mockResolvedValue(page(["a"], 201, 5));
  expect(await fetchEventsPage({ offset: 5 }, signal)).toMatchObject({ total: 201, offset: 5, limit: 200, items: [{ id: "a", provinceCode: "cebu" }] });
  expect(get.mock.calls[0][1]?.signal).toBe(signal);
});
it("fetches the province that exists only after row 200", async () => {
  const first = page(Array.from({ length: 200 }, (_, i) => String(i)), 201);
  const second = page(["last"], 201, 200); second.items[0].provinceCode = "bohol";
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(first).mockResolvedValueOnce(second);
  const result = await fetchPublicEventsCatalog();
  expect(result).toHaveLength(201); expect(result[200].provinceCode).toBe("bohol");
  expect(get.mock.calls.map(([url]) => new URLSearchParams(url.split("?")[1]).get("offset"))).toEqual(["0", "200"]);
});
it("advances by returned length even when the server returns a short page", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(page(["a"], 2)).mockResolvedValueOnce(page(["b"], 2, 1));
  expect((await fetchPublicEventsCatalog()).map(e => e.id)).toEqual(["a", "b"]);
  expect(get.mock.calls[1][0]).toContain("offset=1");
});
it("accepts a truly empty catalog", async () => {
  vi.spyOn(api, "get").mockResolvedValue(page([], 0));
  expect(await fetchPublicEventsCatalog()).toEqual([]);
});
it("rejects a later-page error without returning the first page", async () => {
  vi.spyOn(api, "get").mockResolvedValueOnce(page(["a"], 2)).mockRejectedValueOnce(new Error("Page unavailable"));
  await expect(fetchPublicEventsCatalog()).rejects.toThrow("Page unavailable");
});
it.each(["empty", "offset"])("rejects %s non-progress before total", async kind => {
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(page(["a"], 2)).mockResolvedValue(page(kind === "empty" ? [] : ["b"], 2, kind === "empty" ? 1 : 0));
  await expect(fetchPublicEventsCatalog()).rejects.toThrow(/refresh/i);
  expect(get).toHaveBeenCalledTimes(2);
});
it.each(["total", "overlap"])("restarts once from zero on %s movement", async kind => {
  const get = vi.spyOn(api, "get").mockResolvedValueOnce(page(["a"], 2)).mockResolvedValueOnce(page([kind === "overlap" ? "a" : "b"], kind === "total" ? 3 : 2, 1)).mockResolvedValueOnce(page(["b", "c"], 2));
  expect((await fetchPublicEventsCatalog()).map(e => e.id)).toEqual(["b", "c"]);
  expect(get.mock.calls.map(([url]) => new URLSearchParams(url.split("?")[1]).get("offset"))).toEqual(["0", "1", "0"]);
});
it("rejects continued movement instead of looping or publishing duplicate IDs", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValue(page(["a", "a"], 2));
  await expect(fetchPublicEventsCatalog()).rejects.toThrow(/refresh/i);
  expect(get).toHaveBeenCalledTimes(2);
});
it("stops cancellation between pages without starting another request", async () => {
  const controller = new AbortController();
  const get = vi.spyOn(api, "get").mockImplementation(async () => { controller.abort(); return page(["a"], 2); });
  await expect(fetchPublicEventsCatalog(controller.signal)).rejects.toMatchObject({ name: "AbortError" });
  expect(get).toHaveBeenCalledTimes(1);
});
