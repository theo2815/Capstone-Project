import { render, renderHook, screen, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { useSalesByEvent } from "./admin-sales";
import { AdminSalesLeaderboard } from "@/components/admin/admin-sales-leaderboard";
import { AdminSalesTrend } from "@/components/admin/admin-sales-trend";

vi.mock("next/link", () => ({ default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a> }));

const serverRow = { id: "ev-1", slug: "cebu-run", name: "Cebu City Run", date: "2026-11-01", city: "Cebu", status: "published", state: "past", photoCount: 40, impliedGmv: 3000, impliedCut: 750, refundsIssued: 0 };
const catalogEvent = { ...serverRow, id: "ev-2", name: "Mandaue Night Run" };
const state: { live: (typeof serverRow)[] | null } = { live: null };

vi.mock("@/hooks/use-admin-data", () => ({ useAdminSalesByEventLive: () => state.live }));
vi.mock("@/hooks/use-admin-events", () => ({ useAdminEvents: () => ({ data: [catalogEvent] }), EMPTY_ADMIN_EVENTS: [] }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useAdminDisputes: () => ({ data: [] }), EMPTY_DISPUTES: [] }));
vi.mock("@/hooks/use-admin-payouts", () => ({
  useAdminPayouts: () => ({ data: [{ id: "pc-1", photographerId: "u-ana", photographerName: "Ana Cruz", brandName: "Ana Shoots", handle: "anashoots", weekOf: "2026-11-09", itemCount: 4 }] }),
  EMPTY_PAYOUTS: [],
}));

beforeEach(() => {
  state.live = null;
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

it("top events from the server carry no Implied badge; top photographers carry an Estimate label", () => {
  state.live = [serverRow];
  render(<AdminSalesLeaderboard />);
  expect(screen.queryByText("Implied")).toBeNull();
  const photographers = screen.getByText("Top photographers").closest("div")!;
  expect(within(photographers).getByText("Estimate")).toBeInTheDocument();
  expect(within(photographers).getByText("Estimated from payout photo counts × ₱125.00 until server figures arrive.")).toBeInTheDocument();
});

it("a browser-derived top event keeps its Implied badge", () => {
  render(<AdminSalesLeaderboard />);
  expect(screen.getByText("Implied")).toBeInTheDocument();
});

it("the weekly trend carries an Estimate label", () => {
  render(<AdminSalesTrend />);
  expect(screen.getByText("Estimate")).toBeInTheDocument();
  expect(screen.getByText("Estimated from payout photo counts × ₱125.00 until server figures arrive.")).toBeInTheDocument();
});
