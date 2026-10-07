import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import EarningsPage from "./page";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";

const eventId = "b3ad2fe8-4489-4fbd-9c85-65fe8b3a4c15";
const earning = (id: string, revenueKept: number) => ({
  eventId: id, eventName: id === eventId ? "Cebu Marathon" : id === "free-race" ? "Free Race" : `Race ${id}`,
  eventDate: "2026-09-07", photoCount: 2, salesCount: 3, purchaseCount: 5, revenueKept,
});

function setup(rows: ReturnType<typeof earning>[], total = rows.length, pageItems = (offset: number, limit: number) => rows.slice(offset, offset + limit)) {
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === "/me/photographer/earnings") return {
      lifetimeKept: 100, thisWeek: 0, thisMonth: 0, payoutPending: 0,
      payoutScheduledFor: null, thisWeekSold: 0, thisMonthSold: 0, weeklySeries: [],
    } as never;
    const params = new URLSearchParams(path.split("?")[1]);
    const offset = Number(params.get("offset"));
    const limit = Number(params.get("limit"));
    return { items: pageItems(offset, limit), total, offset, limit } as never;
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><EarningsPage /></QueryClientProvider>);
  return get;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("earnings opens the UUID event with purchased selected", async () => {
  setup([earning(eventId, 100)]);
  const link = await screen.findByRole("link", { name: /Cebu Marathon/ });
  expect(link).toHaveAttribute(
    "href", `/dashboard/events/${eventId}?tab=purchased`,
  );
  expect(link).toHaveTextContent("5 purchases");
  expect(link).toHaveTextContent("2 photos");
  expect(link).toHaveTextContent("₱100.00");
});

it("all earnings pages are reachable, in server order, including zero peso events", async () => {
  const rows = [earning(eventId, 100), ...Array.from({ length: 9 }, (_, i) => earning(`race-${i}`, 90 - i)), earning("free-race", 0)];
  const get = setup(rows);
  expect(await screen.findByRole("link", { name: /Cebu Marathon/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load 1 more" }));
  expect(await screen.findByRole("link", { name: /Free Race/ })).toHaveTextContent("₱0.00");
  const perEventCalls = get.mock.calls.filter(([path]) => String(path).includes("/earnings/per-event?"));
  expect(perEventCalls.map(([path]) => new URLSearchParams(String(path).split("?")[1]).get("offset"))).toEqual(["0", "10"]);
  expect(perEventCalls.map(([path]) => new URLSearchParams(String(path).split("?")[1]).get("limit"))).toEqual(["10", "10"]);
  await waitFor(() => expect(screen.getByText("All 11 loaded")).toBeInTheDocument());
});

it("advances by the consumed server window after a short nonterminal page", async () => {
  const rows = Array.from({ length: 25 }, (_, i) => earning(`race-${i}`, i));
  const get = setup(rows, 25, (offset, limit) => offset === 0 ? rows.slice(0, 2) : rows.slice(offset, offset + limit));
  expect(await screen.findByRole("link", { name: /Race race-0/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load 10 more" }));
  expect(await screen.findByRole("link", { name: /Race race-10/ })).toBeInTheDocument();
  const offsets = get.mock.calls.filter(([path]) => String(path).includes("/earnings/per-event?"))
    .map(([path]) => new URLSearchParams(String(path).split("?")[1]).get("offset"));
  expect(offsets).toEqual(["0", "10"]);
});

it("does not offer another earnings page after an exhausted empty response", async () => {
  const rows = [earning(eventId, 100), earning("race-1", 0)];
  setup(rows, 25, (offset) => offset === 0 ? rows : []);
  expect(await screen.findByRole("link", { name: /Cebu Marathon/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load 10 more" }));
  expect(await screen.findByText("All available event earnings loaded")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Load .* more/ })).toBeNull();
  expect(screen.getByRole("link", { name: /Cebu Marathon/ })).toBeInTheDocument();
});

it("renders overlapping earnings events once in first-seen order without changing server amounts", async () => {
  const rows = [earning("e1", 40), earning("e2", 0), earning("e3", 0)];
  const get = setup(rows, 25, (offset) => offset === 0 ? rows.slice(0, 2)
    : offset === 10 ? [earning("e2", 99), rows[2]] : []);
  expect(await screen.findByRole("link", { name: /Race e1/ })).toHaveTextContent("₱40.00");
  fireEvent.click(screen.getByRole("button", { name: "Load 10 more" }));
  expect(await screen.findByRole("link", { name: /Race e3/ })).toHaveTextContent("₱0.00");
  expect(screen.getAllByRole("link", { name: /Race e2/ })).toHaveLength(1);
  expect(screen.getByRole("link", { name: /Race e2/ })).toHaveTextContent("₱0.00");
  expect(screen.getAllByRole("link", { name: /Race e[123]/ }).map((link) => link.getAttribute("href")))
    .toEqual(["/dashboard/events/e1?tab=purchased", "/dashboard/events/e2?tab=purchased", "/dashboard/events/e3?tab=purchased"]);
  expect(screen.getByText("Showing 3 of 25")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load 10 more" }));
  expect(await screen.findByText("All available event earnings loaded")).toBeInTheDocument();
  expect(get.mock.calls.filter(([path]) => path.includes("/earnings/per-event?"))
    .map(([path]) => new URLSearchParams(path.split("?")[1]).get("offset"))).toEqual(["0", "10", "20"]);
  expect(screen.getByText("Lifetime kept").nextElementSibling).toHaveTextContent("₱100");
});
