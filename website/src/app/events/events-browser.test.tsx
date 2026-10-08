import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { Event } from "@/types/event";
import { EventsBrowser } from "./events-browser";

const navigation = vi.hoisted(() => ({ search: "", listeners: new Set<() => void>(), replace: vi.fn() }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/events",
  useRouter: () => ({ replace: navigation.replace }),
  useSearchParams: () => new URLSearchParams(useSyncExternalStore(cb => { navigation.listeners.add(cb); return () => navigation.listeners.delete(cb); }, () => navigation.search)),
}));
const regions = [
  { code: "region-vii", name: "Central Visayas", shortName: "VII", group: "visayas", provinces: [{ code: "cebu", name: "Cebu" }, { code: "bohol", name: "Bohol" }] },
  { code: "other", name: "Other region", shortName: "Other", group: "luzon", provinces: [{ code: "other-cebu", name: "Cebu" }] },
];
const event = (id: string, pair: Partial<Event> = {}): Event => ({ id, slug: id, name: `Race ${id}`, date: "2099-01-01", location: "Venue, Legacy City", status: "ACTIVE", photoCount: 0, participantCount: 0, regionCode: "region-vii", provinceCode: "cebu", ...pair });
let client: QueryClient;
let rows: Event[];
let referenceFailed: boolean;
let laterFailed: boolean;
let get: ReturnType<typeof vi.spyOn>;
function navigate(search: string) { act(() => { navigation.search = search; navigation.listeners.forEach(cb => cb()); }); }
function mount() { return render(<QueryClientProvider client={client}><EventsBrowser /></QueryClientProvider>); }
async function provinceMenu() { const trigger = await screen.findByRole("button", { name: "Filter by province" }); await waitFor(() => expect(trigger).toBeEnabled()); fireEvent.click(trigger); }
const raceLink = (id: string) => document.querySelector(`a[href="/events/${id}"]`);
beforeEach(() => {
  navigation.search = ""; navigation.replace.mockReset().mockImplementation((url: string) => { navigation.search = url.split("?")[1] ?? ""; navigation.listeners.forEach(cb => cb()); });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  rows = [event("a")]; referenceFailed = false; laterFailed = false;
  get = vi.spyOn(api, "get").mockImplementation(async path => {
    if (path === "/regions") { if (referenceFailed) throw new Error("Reference unavailable"); return regions; }
    if (!path.startsWith("/events?")) throw new Error(`Unexpected ${path}`);
    const offset = Number(new URLSearchParams(path.split("?")[1]).get("offset"));
    if (offset && laterFailed) throw new Error("Later page unavailable");
    return { items: rows.slice(offset, offset + 200), total: rows.length, offset, limit: 200 };
  });
});
afterEach(() => { cleanup(); client.clear(); vi.restoreAllMocks(); });

it("offers and returns the only Bohol event after row 200, including upcoming zero-photo events", async () => {
  rows = [...Array.from({ length: 200 }, (_, i) => event(String(i))), event("last", { provinceCode: "bohol" })];
  mount(); await provinceMenu();
  fireEvent.click(await screen.findByRole("menuitem", { name: "Bohol" }));
  await waitFor(() => expect(raceLink("last")).toBeInTheDocument());
  expect(raceLink("0")).not.toBeInTheDocument();
  expect(screen.getByText("of").parentElement).toHaveTextContent("1 of 201 race");
  expect(navigation.search).toBe("province=bohol");
});
it("validates stored pairs, disambiguates labels, and keeps unclassified events under All provinces", async () => {
  rows = [event("a"), event("other", { regionCode: "other", provinceCode: "other-cebu" }), event("legacy", { regionCode: null, provinceCode: null }), event("invalid", { regionCode: "other", provinceCode: "bohol" })];
  mount(); await provinceMenu();
  expect(await screen.findByRole("menuitem", { name: "Cebu · VII" })).toBeInTheDocument();
  expect(screen.getByRole("menuitem", { name: "Cebu · Other" })).toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Bohol" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("menuitem", { name: "All provinces" }));
  expect(raceLink("legacy")).toBeInTheDocument(); expect(raceLink("invalid")).toBeInTheDocument();
  navigate("province=bohol");
  await screen.findByText("Nothing here yet."); expect(raceLink("invalid")).not.toBeInTheDocument();
});
it.each([false, true])("handles empty/all-unclassified catalog (%s) without fabricated options", async unclassified => {
  rows = unclassified ? [event("legacy", { regionCode: null, provinceCode: null })] : [];
  mount(); await provinceMenu();
  await waitFor(() => expect(client.isFetching()).toBe(0));
  expect(screen.getAllByRole("menuitem")).toHaveLength(1);
  if (unclassified) expect(raceLink("legacy")).toBeInTheDocument(); else expect(screen.getByText("Nothing here yet.")).toBeInTheDocument();
});
it("shows an unknown URL province as invalid and empty without resetting it", async () => {
  navigation.search = "province=unknown"; mount();
  expect(await screen.findByRole("alert")).toHaveTextContent(/unrecognized province/i);
  expect(raceLink("a")).not.toBeInTheDocument(); expect(navigation.search).toBe("province=unknown");
  fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
  await waitFor(() => expect(raceLink("a")).toBeInTheDocument());
});
it("keeps legacy city explicit, replaces it atomically, and lets province win when both arrive", async () => {
  navigation.search = "city=Legacy+City&q=Race"; mount();
  await screen.findByRole("button", { name: "Filter by province" });
  expect(screen.getAllByText(/Legacy city: Legacy City/).length).toBeGreaterThan(0);
  await provinceMenu(); fireEvent.click(await screen.findByRole("menuitem", { name: "Cebu" }));
  await waitFor(() => expect(new URLSearchParams(navigation.search).get("province")).toBe("cebu"));
  expect(new URLSearchParams(navigation.search).has("city")).toBe(false);
  expect(new URLSearchParams(navigation.search).get("q")).toBe("Race");
  navigate("province=bohol&city=Legacy+City");
  await waitFor(() => expect(navigation.search).toBe("province=bohol"));
  expect(raceLink("a")).not.toBeInTheDocument();
});
it("keeps options independent of search/date, merges debounced search, and follows external navigation", async () => {
  rows.push(event("last", { provinceCode: "bohol" })); mount();
  await waitFor(() => expect(raceLink("a")).toBeInTheDocument());
  fireEvent.change(screen.getByRole("searchbox"), { target: { value: "last" } });
  await provinceMenu(); fireEvent.click(screen.getByRole("menuitem", { name: "Bohol" }));
  await waitFor(() => expect(new URLSearchParams(navigation.search).get("q")).toBe("last"));
  expect(new URLSearchParams(navigation.search).get("province")).toBe("bohol");
  fireEvent.click(screen.getByRole("button", { name: "Filter by date" })); fireEvent.click(screen.getByRole("menuitem", { name: "Archive" }));
  await screen.findByText("Nothing here yet."); await provinceMenu();
  expect(screen.getByRole("menuitem", { name: "Cebu" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("menuitem", { name: "All provinces" }));
  fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
  await waitFor(() => expect(navigation.search).toBe("q=last"));
  navigate("province=cebu"); await waitFor(() => expect(raceLink("a")).toBeInTheDocument());
  expect(screen.getByRole("searchbox")).toHaveValue(""); expect(raceLink("last")).not.toBeInTheDocument();
});
it.each(["reference", "later"])("shows %s failure and retries without claiming an empty complete catalog", async kind => {
  rows = [...Array.from({ length: 200 }, (_, i) => event(String(i))), event("last", { provinceCode: "bohol" })];
  referenceFailed = kind === "reference"; laterFailed = kind === "later";
  mount(); expect(await screen.findByRole("alert")).toHaveTextContent(kind === "reference" ? /province/i : /events/i);
  expect(screen.queryByText("Nothing here yet.")).not.toBeInTheDocument();
  referenceFailed = false; laterFailed = false;
  fireEvent.click(screen.getByRole("button", { name: kind === "reference" ? "Retry provinces" : "Retry events" }));
  await provinceMenu(); fireEvent.click(await screen.findByRole("menuitem", { name: "Bohol" }));
  await waitFor(() => expect(raceLink("last")).toBeInTheDocument());
});
it("keeps only the last complete catalog on refresh failure and removes a deleted province without clearing selection", async () => {
  rows = [event("a"), event("last", { provinceCode: "bohol" })]; navigation.search = "province=bohol";
  mount(); await waitFor(() => expect(raceLink("last")).toBeInTheDocument());
  rows = [...Array.from({ length: 201 }, (_, i) => event(String(i)))]; laterFailed = true;
  await act(async () => { await client.invalidateQueries({ queryKey: ["public", "events"] }); });
  expect(await screen.findByRole("alert")).toHaveTextContent(/events/i);
  expect(raceLink("last")).toBeInTheDocument(); expect(raceLink("0")).not.toBeInTheDocument();
  rows = [event("a")]; laterFailed = false;
  fireEvent.click(screen.getByRole("button", { name: "Retry events" }));
  await screen.findByText("Nothing here yet."); expect(navigation.search).toBe("province=bohol");
  await provinceMenu(); expect(screen.queryByRole("menuitem", { name: "Bohol" })).not.toBeInTheDocument();
});
it("never treats a one-page query cache as a complete catalog", async () => {
  client.setQueryData(["public", "events", {}], []); mount();
  await waitFor(() => expect(raceLink("a")).toBeInTheDocument());
  expect(get).toHaveBeenCalledWith(expect.stringContaining("/events?"), expect.anything());
  expect(client.getQueryData(["public", "events", {}])).toEqual([]);
});
it("preserves grouped lifecycle counts and Load More across the complete catalog", async () => {
  rows = [...Array.from({ length: 201 }, (_, i) => event(String(i))), event("archive", { date: "2000-01-01" })];
  mount();
  expect(await screen.findByText("201 races")).toBeInTheDocument();
  expect(document.querySelector("#archive")).toHaveTextContent("1 race");
  const before = document.querySelectorAll("#upcoming a[href^='/events/']").length;
  fireEvent.click(screen.getByRole("button", { name: /Load .* more/ }));
  expect(document.querySelectorAll("#upcoming a[href^='/events/']").length).toBeGreaterThan(before);
  await provinceMenu(); fireEvent.click(screen.getByRole("menuitem", { name: "Cebu" }));
  await waitFor(() => expect(screen.getByText("of").parentElement).toHaveTextContent("202 of 202 races"));
  expect(document.querySelectorAll("a[href^='/events/']")).toHaveLength(before);
});
it("removes even the legacy all sentinel when province and city arrive together", async () => {
  navigation.search = "province=cebu&city=all"; mount();
  await waitFor(() => expect(navigation.search).toBe("province=cebu"));
});
