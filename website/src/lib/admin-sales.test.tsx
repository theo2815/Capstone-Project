import { render, renderHook, screen, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import * as sales from "./admin-sales";
import { useSalesByEvent } from "./admin-sales";
import { fetchAdminSalesTrend, fetchAdminTopPhotographers, type AdminTopPhotographer } from "./api-admin";
import { AdminSalesLeaderboard } from "@/components/admin/admin-sales-leaderboard";
import { AdminSalesEventTable } from "@/components/admin/admin-sales-event-table";
import { AdminSalesTrend } from "@/components/admin/admin-sales-trend";
import { formatLongDate } from "@/lib/format";

vi.mock("next/link", () => ({ default: ({ children, href, className }: { children: React.ReactNode; href: string; className?: string }) => <a href={href} className={className}>{children}</a> }));

const apiGet = vi.hoisted(() => vi.fn());
vi.mock("@/lib/api", () => ({ api: { get: apiGet }, ApiError: class extends Error {} }));

const serverRow = { id: "ev-1", slug: "cebu-run", name: "Cebu City Run", date: "2026-11-01", city: "Cebu", status: "published", state: "past", photoCount: 40, impliedGmv: 3000, impliedCut: 750, refundsIssued: 0 };
const catalogEvent = { ...serverRow, id: "ev-2", name: "Mandaue Night Run" };
const weeks = [
  { weekStart: "2026-11-09", gmv: 1000, platformRevenue: 200, refundsIssued: 0, salesCount: 8 },
  { weekStart: "2026-11-16", gmv: 2500, platformRevenue: 500, refundsIssued: 0, salesCount: 20 },
];
const top: AdminTopPhotographer[] = [{ userId: "u-ana", handle: "anashoots", displayName: "Ana Shoots", gmv: 4200, salesCount: 33 }];
type Q<T> = { data?: T; isError: boolean; isFetching: boolean; refetch: () => void };
const state: {
  live: (typeof serverRow)[] | null;
  trend: Q<typeof weeks>;
  top: Q<typeof top>;
  topArgs: unknown[];
} = { live: null, trend: { isError: false, isFetching: false, refetch: vi.fn() }, top: { isError: false, isFetching: false, refetch: vi.fn() }, topArgs: [] };
const payouts = vi.hoisted(() => vi.fn(() => ({ data: [] })));

vi.mock("@/hooks/use-admin-data", () => ({
  useAdminSalesByEventLive: () => state.live,
  useAdminSalesTrendQuery: () => state.trend,
  useAdminTopPhotographersQuery: (...args: unknown[]) => {
    state.topArgs = args;
    return state.top;
  },
}));
vi.mock("@/hooks/use-admin-events", () => ({ useAdminEvents: () => ({ data: [catalogEvent] }), EMPTY_ADMIN_EVENTS: [] }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useAdminDisputes: () => ({ data: [] }), EMPTY_DISPUTES: [] }));
vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: payouts, EMPTY_PAYOUTS: [] }));

beforeEach(() => {
  state.live = null;
  state.trend = { data: weeks, isError: false, isFetching: false, refetch: vi.fn() };
  state.top = { data: top, isError: false, isFetching: false, refetch: vi.fn() };
  apiGet.mockReset();
  payouts.mockClear();
});

it("server by-event rows are real sums, not implied", () => {
  state.live = [serverRow];
  const { result } = renderHook(() => useSalesByEvent());
  expect(result.current).toEqual([expect.objectContaining({ id: "ev-1", implied: false })]);
});

it("rows derived in the browser while the server has no answer are implied", () => {
  const { result } = renderHook(() => useSalesByEvent());
  expect(result.current).toEqual([expect.objectContaining({ id: "ev-2", implied: true })]);
});

it("top photographers come from the server list, year to date, with no Estimate label", () => {
  state.live = [serverRow];
  render(<AdminSalesLeaderboard />);
  const photographers = screen.getByText("Top photographers").closest("div")!;
  expect(within(photographers).getByText("Ana Shoots").closest("a")).toHaveAttribute("href", "/admin/photographers/anashoots");
  expect(within(photographers).getByText("₱4,200.00")).toBeInTheDocument();
  expect(within(photographers).getByText("33 photos sold")).toBeInTheDocument();
  expect(state.topArgs).toEqual(["ytd", 10]);
  expect(screen.queryByText("Estimate")).toBeNull();
  expect(screen.queryByText(/Estimated from payout/)).toBeNull();
  expect(screen.queryByText("Implied")).toBeNull();
});

it("a failed top-photographers request says so instead of showing an empty board", () => {
  state.top = { isError: true, isFetching: false, refetch: vi.fn() };
  render(<AdminSalesLeaderboard />);
  const alert = screen.getByRole("alert");
  within(alert).getByRole("button", { name: "Retry" }).click();
  expect(state.top.refetch).toHaveBeenCalled();
  expect(screen.queryByText(/No photo sales/)).toBeNull();
});

it("a photographer with no handle links to their own page by user id", () => {
  state.top = { ...state.top, data: [{ ...top[0], userId: "u-nora", handle: null, displayName: "Nora Lim" }] };
  render(<AdminSalesLeaderboard />);
  expect(screen.getByText("Nora Lim").closest("a")).toHaveAttribute("href", "/admin/photographers/u-nora");
});

it("long photographer and event names wrap instead of being cut", () => {
  const long = "Mandaue Southern Visayas Endurance Photography Collective Co.";
  state.top = { ...state.top, data: [{ ...top[0], displayName: long }] };
  state.live = [{ ...serverRow, name: long }];
  render(<AdminSalesLeaderboard />);
  expect(screen.getAllByText(long)).toHaveLength(2);
  for (const name of screen.getAllByText(long)) {
    expect(name).not.toHaveClass("truncate");
    expect(name).toHaveClass("break-words");
  }
});

it("a long event name in Sales by event wraps instead of being cut", () => {
  const long = "Mandaue Southern Visayas Endurance Marathon and Fun Run 2026";
  state.live = [{ ...serverRow, name: long }];
  render(<AdminSalesEventTable />);
  const name = screen.getByText(long);
  expect(name).not.toHaveClass("truncate");
  expect(name).toHaveClass("break-words");
});

it("the by-event headings and every row share one set of columns", () => {
  state.live = [serverRow, { ...serverRow, id: "ev-3", impliedGmv: 12345678.9 }];
  const { container } = render(<AdminSalesEventTable />);
  // One element sizes the columns; the headings and each row are subgrids of it, so a big figure widens its
  // column for every row at once. jsdom has no layout: the browser check measures the alignment.
  const templates = [...container.querySelectorAll("*")].filter((el) => el.getAttribute("class")?.includes("lg:grid-cols-["));
  expect(templates).toHaveLength(1);
  const subgrids = [...container.querySelectorAll(".lg\\:grid-cols-subgrid")];
  for (const el of subgrids) {
    expect(el).toHaveClass("lg:grid", "lg:col-span-full");
    expect(el.parentElement === templates[0] || subgrids.includes(el.parentElement!)).toBe(true);
  }
  expect(subgrids).toContain(screen.getByText("Event").parentElement);
  for (const gmv of [screen.getByText("₱3,000.00"), screen.getByText("₱12,345,678.90")]) {
    expect(subgrids).toContain(gmv.parentElement!.parentElement);
  }
});

it("from 1024 up, a refund and a date in Sales by event stay on one line", () => {
  state.live = [{ ...serverRow, refundsIssued: 120 }];
  render(<AdminSalesEventTable />);
  // jsdom has no layout: the browser check measures one line at 1024 and 1280.
  for (const value of [screen.getByText("−₱120.00"), screen.getByText(formatLongDate(serverRow.date, true))]) {
    expect(value).toHaveClass("lg:whitespace-nowrap");
  }
});

it("an event name can shrink to its column, so a long word breaks there instead of running into the date", () => {
  state.live = [serverRow];
  render(<AdminSalesEventTable />);
  // The name is a flex item: without min-w-0 it cannot get narrower than its longest word (browser check: 72 px column).
  expect(screen.getByText(serverRow.name)).toHaveClass("min-w-0", "break-words");
});

it("a browser-derived top event keeps its Estimate badge", () => {
  render(<AdminSalesLeaderboard />);
  expect(screen.getByText("Estimate")).toBeInTheDocument();
});

it("the weekly trend draws the server weeks, with no Estimate label", () => {
  render(<AdminSalesTrend />);
  expect(screen.getByText("₱3,500.00 total")).toBeInTheDocument();
  expect(screen.getByText("Weekly total sales · last 2 weeks")).toBeInTheDocument();
  expect(screen.queryByText("Estimate")).toBeNull();
  expect(screen.queryByText(/Estimated from payout/)).toBeNull();
});

it("the weekly trend shows a skeleton while loading and Retry on failure, never zero bars", () => {
  state.trend = { isError: false, isFetching: true, refetch: vi.fn() };
  const { container, unmount } = render(<AdminSalesTrend />);
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(screen.queryByText(/total/)).toBeNull();
  unmount();
  state.trend = { isError: true, isFetching: false, refetch: vi.fn() };
  render(<AdminSalesTrend />);
  within(screen.getByRole("alert")).getByRole("button", { name: "Retry" }).click();
  expect(state.trend.refetch).toHaveBeenCalled();
});

it("the estimate helpers are gone and nothing reads payout cycles for sales", () => {
  render(<><AdminSalesTrend /><AdminSalesLeaderboard /></>);
  expect(payouts).not.toHaveBeenCalled();
  expect(sales).not.toHaveProperty("SALES_ESTIMATE_NOTE");
  expect(sales).not.toHaveProperty("useWeeklyGmvSeries");
  expect(sales).not.toHaveProperty("useTopPhotographers");
});

it("the fetchers call the AX-12 routes and unwrap their lists", async () => {
  apiGet.mockResolvedValueOnce({ weeks }).mockResolvedValueOnce({ items: top });
  await expect(fetchAdminSalesTrend(12)).resolves.toEqual(weeks);
  await expect(fetchAdminTopPhotographers("ytd", 10)).resolves.toEqual(top);
  expect(apiGet.mock.calls.map((c) => c[0])).toEqual([
    "/admin/sales/trend?weeks=12",
    "/admin/sales/top-photographers?range=ytd&limit=10",
  ]);
});

it("an event row in either list opens the Events page searched for that event, not a missing page (FT-4)", () => {
  state.live = [{ ...serverRow, name: "Cebu Marathon" }];
  for (const List of [AdminSalesEventTable, AdminSalesLeaderboard]) {
    const { unmount } = render(<List />);
    expect(screen.getByText("Cebu Marathon").closest("a")).toHaveAttribute("href", "/admin/events?q=Cebu%20Marathon");
    unmount();
  }
});

it("a week with no sales shows a sentence, not an empty grey chart (FT-26)", () => {
  state.trend = { ...state.trend, data: weeks.map((w) => ({ ...w, gmv: 0 })) };
  render(<AdminSalesTrend />);
  expect(screen.getByText("No sales in the last 2 weeks.")).toBeInTheDocument();
  expect(screen.queryAllByRole("button")).toHaveLength(0);
});

it("Sales by event uses plain names for its columns, sort and estimate badge (FT-26)", () => {
  render(<AdminSalesEventTable />);
  expect(screen.getByText("Sort by")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Total sales ↓" })).toBeInTheDocument();
  expect(screen.getAllByText("Total sales").length).toBeGreaterThanOrEqual(2);
  expect(screen.getAllByText("Platform cut").length).toBeGreaterThanOrEqual(2);
  expect(screen.getByText("Estimate")).toBeInTheDocument();
  for (const gone of [/GMV/, "Cut", "Implied"]) expect(screen.queryByText(gone)).toBeNull();
});

it("top events say what they are ranked by and over which period (FT-26)", () => {
  render(<AdminSalesLeaderboard />);
  expect(screen.getByText("By total sales · all time")).toBeInTheDocument();
  expect(screen.queryByText(/gross merchandise/)).toBeNull();
});
