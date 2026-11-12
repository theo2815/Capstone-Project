import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { AdminKpiStrip } from "./admin-kpi-strip";

vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: () => ({ data: [] }), EMPTY_PAYOUTS: [] }));
vi.mock("@/hooks/use-admin-users", () => ({ useAdminUsers: () => ({ data: [] }), EMPTY_ADMIN_USERS: [] }));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminFlags: () => ({ total: 0 }),
  useAdminKpis: () => ({
    pendingVerifications: 0, approvedPhotographers: 0, suspended: 0, liveEvents: 0,
    decisionsThisWeek: 0, openDisputes: 1, disputesNeedingAdmin: 5, disputesOverdue: 2,
    openFlags: 0, pendingPayouts: 0, pendingEventRequests: 0,
  }),
}));

afterEach(cleanup);

it("the strip shows the server's needs-you count", () => {
  render(<AdminKpiStrip />);
  expect(screen.getByRole("link", { name: "5 disputes" })).toBeTruthy();
});
