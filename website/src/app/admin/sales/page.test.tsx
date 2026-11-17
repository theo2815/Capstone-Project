import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import AdminSalesPage from "./page";

vi.mock("@/components/admin/admin-sales-trend", () => ({ AdminSalesTrend: () => null }));
vi.mock("@/components/admin/admin-sales-leaderboard", () => ({ AdminSalesLeaderboard: () => null }));
vi.mock("@/components/admin/admin-sales-event-table", () => ({ AdminSalesEventTable: () => null }));
vi.mock("@/lib/admin-sales", () => ({
  useSalesKpis: () => ({ gmv: 48000, platformRevenue: 9600, refundsIssued: 120, netPlatformRevenue: 9480, photographerKeep: 38400, totalSalesCount: 400 }),
  // The dispute list has not loaded any refunded rows yet, while the server total says ₱120.
  useRefundPulse: () => ({ refundsIssued: 0, refundCount: 0, refundRatePct: 0, openCount: 0, escalatedCount: 0 }),
}));

it("never captions a non-zero refund total with 'no refunds processed'", () => {
  render(<AdminSalesPage />);
  const tile = screen.getByText("₱120.00").parentElement!;
  expect(tile).toHaveTextContent("₱120.00");
  expect(tile).toHaveTextContent("refunded to runners");
  expect(tile).not.toHaveTextContent("no refunds processed");
});
