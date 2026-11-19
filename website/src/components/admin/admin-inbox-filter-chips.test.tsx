import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";
import { ADMIN_QUEUES } from "@/lib/admin-queues";
import { AdminInboxFilterChips } from "./admin-inbox-filter-chips";

const replace = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace, push: vi.fn() }),
  usePathname: () => "/admin/inbox",
  useSearchParams: () => new URLSearchParams("type=removals&offset=20&eventId=x"),
}));
const kpis = vi.hoisted(() => ({ value: null as AdminKpis | null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));

afterEach(() => { cleanup(); replace.mockReset(); kpis.value = null; });

it("shows one chip per queue, in the shared order, with its KPI count", () => {
  kpis.value = {
    pendingVerifications: 3, approvedPhotographers: 0, suspended: 0, liveEvents: 0, decisionsThisWeek: 0,
    openDisputes: 1, disputesNeedingAdmin: 5, openFlags: 0, pendingPayouts: 9, pendingEventRequests: 2,
    pendingSellerDetails: 4, payoutsAwaitingReview: 6, openPayoutReports: 1, relevanceNeedsPerson: 7, removalsNeedingAction: 8,
  };
  render(<AdminInboxFilterChips value="verifications" />);
  const chips = screen.getAllByRole("button");
  expect(chips.map((c) => c.textContent?.replace(/\d+$/, "").trim())).toEqual(ADMIN_QUEUES.map((q) => q.label));
  expect(screen.getByRole("button", { name: "Photo removals 8" })).toBeTruthy();
  expect(screen.getByRole("button", { name: "Payouts 6" })).toBeTruthy();
  expect(screen.getByRole("button", { name: /^Verifications/ })).toHaveAttribute("aria-pressed", "true");
});

it("shows no count for a queue the backend does not send", () => {
  kpis.value = {
    pendingVerifications: 3, approvedPhotographers: 0, suspended: 0, liveEvents: 0, decisionsThisWeek: 0,
    openDisputes: 1, openFlags: 0, pendingPayouts: 9, pendingEventRequests: 2,
  };
  render(<AdminInboxFilterChips value="verifications" />);
  expect(screen.getByRole("button", { name: "Photo removals" })).toBeTruthy();
});

it("switching chips drops every other list's params, leaving only ?type=<new>", () => {
  render(<AdminInboxFilterChips value="removals" />);
  fireEvent.click(screen.getByRole("button", { name: /^Disputes/ }));
  expect(replace).toHaveBeenCalledTimes(1);
  expect(replace).toHaveBeenCalledWith("/admin/inbox?type=disputes", { scroll: false });
});
