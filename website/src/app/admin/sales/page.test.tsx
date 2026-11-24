import { render, screen, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import AdminSalesPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("@/components/admin/admin-sales-trend", () => ({ AdminSalesTrend: () => null }));
vi.mock("@/components/admin/admin-sales-leaderboard", () => ({ AdminSalesLeaderboard: () => null }));
vi.mock("@/components/admin/admin-sales-event-table", () => ({ AdminSalesEventTable: () => null }));

const ytd = { gmv: 48000, platformRevenue: 9600, refundsIssued: 120, netPlatformRevenue: 9480, photographerKeep: 38400, totalSalesCount: 400 };
const week = { gmv: 2000, platformRevenue: 400, refundsIssued: 50, netPlatformRevenue: 350, photographerKeep: 1600, totalSalesCount: 16 };
type Q = { data?: typeof ytd; isError: boolean; isFetching: boolean; refetch: () => void };
const state: { ytd: Q; week: Q } = { ytd: { isError: false, isFetching: false, refetch: vi.fn() }, week: { isError: false, isFetching: false, refetch: vi.fn() } };

vi.mock("@/hooks/use-admin-data", () => ({
  useAdminSalesKpisQuery: (range: "ytd" | "week") => state[range],
}));
vi.mock("@/lib/admin-sales", () => ({
  // The dispute list has not loaded any refunded rows yet, while the server total says ₱120.
  useRefundPulse: () => ({ refundCount: 0, openCount: 0, escalatedCount: 0 }),
}));

beforeEach(() => {
  state.ytd = { data: ytd, isError: false, isFetching: false, refetch: vi.fn() };
  state.week = { data: week, isError: false, isFetching: false, refetch: vi.fn() };
});

it("never captions a non-zero refund total with 'no refunds processed'", () => {
  render(<AdminSalesPage />);
  const tile = screen.getAllByText("₱120.00")[0].parentElement!;
  expect(tile).toHaveTextContent("refunded to runners");
  expect(tile).not.toHaveTextContent("no refunds processed");
});

it("shows a skeleton, not figures, while the money loads", () => {
  state.ytd = { isError: false, isFetching: true, refetch: vi.fn() };
  const { container } = render(<AdminSalesPage />);
  expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();
  expect(screen.queryByText(/₱/)).toBeNull();
});

it("a failed money query shows an alert with Retry and no derived money", () => {
  state.ytd = { isError: true, isFetching: false, refetch: vi.fn() };
  state.week = { isError: true, isFetching: false, refetch: vi.fn() };
  render(<AdminSalesPage />);
  const alert = screen.getByRole("alert");
  within(alert).getByRole("button", { name: "Retry" }).click();
  expect(state.ytd.refetch).toHaveBeenCalled();
  expect(screen.queryByText(/₱/)).toBeNull();
  expect(screen.queryByText("no refunds processed")).toBeNull();
  expect(screen.getByText(/photos sold so far this year/)).toHaveTextContent("— photos sold so far this year.");
});

it("the refund pulse agrees with the refunds tile and rates against this week's GMV", () => {
  render(<AdminSalesPage />);
  expect(screen.getAllByText("₱120.00")).toHaveLength(2);
  expect(screen.getByText("2.50%")).toBeInTheDocument();
  expect(screen.queryByText("0 refunds processed")).toBeNull();
  expect(screen.getByText("refunded to runners, year to date")).toBeInTheDocument();
});

it("shows no refund rate for a week without sales, even when an older sale was refunded", () => {
  state.week = { data: { ...week, gmv: 0, refundsIssued: 50 }, isError: false, isFetching: false, refetch: vi.fn() };
  render(<AdminSalesPage />);
  expect(screen.queryByText("0.00%")).toBeNull();
  expect(screen.getByText("of this week's total sales").closest("li")).toHaveTextContent("—");
});

it("reads the year's photo count from its value, not fixed copy", () => {
  render(<AdminSalesPage />);
  expect(screen.getByText(/photos sold so far this year/)).toHaveTextContent("400 photos sold so far this year.");
});

it("the heading is the rail label", () => {
  render(<AdminSalesPage />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Sales.");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  render(<AdminSalesPage />);
  expectSectionIntro(ROUTES.ADMIN_SALES);
});

it("the open refunds line says they wait for a decision", () => {
  render(<AdminSalesPage />);
  expect(screen.getByText("waiting for a decision")).toBeInTheDocument();
});

it("every money tile says its period, and the platform's share has one name (FT-26)", () => {
  render(<AdminSalesPage />);
  // The guide's definitions also name these terms, so find each tile by its caption.
  expect(screen.getByText("400 photos sold · year to date").parentElement).toHaveTextContent(/^Total sales₱48,000\.00/);
  expect(screen.getByText("QuickPitik's share after refunds · year to date").parentElement).toHaveTextContent(/^Platform cut₱9,600\.00/);
  expect(screen.getAllByText("refunded to runners · year to date")).toHaveLength(1);
  // The backend's net figure is the cut by construction, so there is no fourth tile.
  expect(screen.queryByText("Net platform revenue")).toBeNull();
  expect(screen.queryByText("Platform cut after refunds")).toBeNull();
  for (const gone of [/Gross merch/, /Platform revenue/, /GMV/, /% of every sale/]) expect(screen.queryByText(gone)).toBeNull();
});

it("the section captions name their period in plain words (FT-26)", () => {
  render(<AdminSalesPage />);
  for (const caption of ["Year to date", "Total sales per week", "Sortable · estimates are marked", "This week and year to date"]) {
    expect(screen.getByText(caption)).toBeInTheDocument();
  }
});

it("the refund pulse names the open disputes in words (FT-26)", () => {
  render(<AdminSalesPage />);
  const row = screen.getByText("Disputes still open").closest("li")!;
  expect(row).toHaveTextContent("0 open, 0 escalated");
  expect(screen.queryByText("Open · escalated")).toBeNull();
});
