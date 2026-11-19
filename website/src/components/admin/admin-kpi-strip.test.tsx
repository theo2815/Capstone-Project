import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";
import { AdminKpiStrip } from "./admin-kpi-strip";

const kpis = vi.hoisted(() => ({ value: null as AdminKpis | null, failed: false }));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminKpis: () => kpis.value,
  useAdminKpisQuery: () => ({ data: kpis.value ?? undefined, isPending: kpis.value === null && !kpis.failed }),
}));
// Counts worked out in the browser from other lists are gone (IN4).
vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: () => ({ data: [{ status: "pending_review" }] }), EMPTY_PAYOUTS: [] }));
vi.mock("@/hooks/use-admin-users", () => ({
  useAdminUsers: () => ({ data: [{ role: "PHOTOGRAPHER", verificationStatus: "pending", suspendedAt: null }] }),
  EMPTY_ADMIN_USERS: [],
}));

const base: AdminKpis = {
  pendingVerifications: 0, approvedPhotographers: 0, suspended: 0, liveEvents: 0,
  decisionsThisWeek: 0, openDisputes: 1, disputesNeedingAdmin: 5, disputesOverdue: 2,
  openFlags: 0, pendingPayouts: 0, pendingEventRequests: 0,
};

afterEach(() => { cleanup(); kpis.value = null; kpis.failed = false; });

it("leaves the strip out when the KPI request failed, instead of loading forever", () => {
  kpis.failed = true;
  render(<AdminKpiStrip />);
  expect(screen.queryByLabelText("Queue counts")).not.toBeInTheDocument();
});

it("the strip shows the server's needs-you count", () => {
  kpis.value = base;
  render(<AdminKpiStrip />);
  expect(screen.getByRole("link", { name: "5 disputes" })).toBeTruthy();
});

it("shows no numbers while the KPIs load", () => {
  render(<AdminKpiStrip />);
  expect(screen.queryAllByRole("link")).toHaveLength(0);
  expect(screen.getByLabelText("Queue counts")).toHaveAttribute("aria-busy", "true");
  expect(screen.getByLabelText("Queue counts").textContent).not.toMatch(/\d/);
});

it("lists the queues in the shared order, skipping fields the backend lacks", () => {
  kpis.value = { ...base, payoutsAwaitingReview: 1, removalsNeedingAction: 2 };
  render(<AdminKpiStrip />);
  expect(screen.getAllByRole("link").map((l) => l.getAttribute("aria-label"))).toEqual([
    "5 disputes", "0 flags", "0 verifications", "0 event requests", "1 payout", "2 photo removals"]);
  for (const link of screen.getAllByRole("link")) expect(link.className).toContain("min-h-[44px]");
});
