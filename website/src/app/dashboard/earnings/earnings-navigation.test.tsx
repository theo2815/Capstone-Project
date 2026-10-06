import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import EarningsPage from "./page";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";

const eventId = "b3ad2fe8-4489-4fbd-9c85-65fe8b3a4c15";
const earning = (id: string, revenueKept: number) => ({
  eventId: id, eventName: id === eventId ? "Cebu Marathon" : id === "free-race" ? "Free Race" : `Race ${id}`,
  eventDate: "2026-09-07", photoCount: 2, salesCount: 3, revenueKept,
});

function setup(rows: ReturnType<typeof earning>[], total = rows.length) {
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (path === "/me/photographer/earnings") return {
      lifetimeKept: 100, thisWeek: 0, thisMonth: 0, payoutPending: 0,
      payoutScheduledFor: null, thisWeekSold: 0, thisMonthSold: 0, weeklySeries: [],
    } as never;
    const params = new URLSearchParams(path.split("?")[1]);
    const offset = Number(params.get("offset"));
    const limit = Number(params.get("limit"));
    return { items: rows.slice(offset, offset + limit), total, offset, limit } as never;
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><EarningsPage /></QueryClientProvider>);
  return get;
}

afterEach(() => { cleanup(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("earnings opens the UUID event with purchased selected", async () => {
  setup([earning(eventId, 100)]);
  expect(await screen.findByRole("link", { name: /Cebu Marathon/ })).toHaveAttribute(
    "href", `/dashboard/events/${eventId}?tab=purchased`,
  );
});

it("all earnings pages are reachable, in server order, including zero peso events", async () => {
  const rows = [earning(eventId, 100), ...Array.from({ length: 9 }, (_, i) => earning(`race-${i}`, 90 - i)), earning("free-race", 0)];
  const get = setup(rows);
  expect(await screen.findByRole("link", { name: /Cebu Marathon/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Load 1 more" }));
  expect(await screen.findByRole("link", { name: /Free Race/ })).toBeInTheDocument();
  const perEventCalls = get.mock.calls.filter(([path]) => String(path).includes("/earnings/per-event?"));
  expect(perEventCalls.map(([path]) => new URLSearchParams(String(path).split("?")[1]).get("offset"))).toEqual(["0", "10"]);
  expect(perEventCalls.map(([path]) => new URLSearchParams(String(path).split("?")[1]).get("limit"))).toEqual(["10", "10"]);
  await waitFor(() => expect(screen.getByText("All 11 loaded")).toBeInTheDocument());
});
