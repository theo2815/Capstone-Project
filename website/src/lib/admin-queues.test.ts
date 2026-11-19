import { renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";
import { ADMIN_QUEUES, useAdminAttentionTarget, useAdminQueueCounts } from "./admin-queues";

const kpis = vi.hoisted(() => ({ value: null as AdminKpis | null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));

// A backend from before AX-8: none of the new queue fields.
const base: AdminKpis = {
  pendingVerifications: 0, approvedPhotographers: 0, suspended: 0, liveEvents: 0,
  decisionsThisWeek: 0, openDisputes: 1, openFlags: 0, pendingPayouts: 0, pendingEventRequests: 0,
};
const full: AdminKpis = {
  ...base, disputesNeedingAdmin: 1, openFlags: 2, pendingVerifications: 3, pendingSellerDetails: 4,
  pendingEventRequests: 5, payoutsAwaitingReview: 6, pendingPayouts: 60, openPayoutReports: 7,
  relevanceNeedsPerson: 8, relevanceHidden: 80, removalsNeedingAction: 9,
};

afterEach(() => { kpis.value = null; });

it("lists the queues in one order", () => {
  expect(ADMIN_QUEUES.map((q) => q.label)).toEqual(["Disputes", "Flags", "Verifications", "Seller details",
    "Event requests", "Payouts", "Payout reports", "Photo Relevance", "Photo removals"]);
});

it("reads every count from its KPI field", () => {
  kpis.value = full;
  const { byKey, total } = renderHook(() => useAdminQueueCounts()).result.current;
  expect(byKey).toEqual({ disputes: 1, flags: 2, verifications: 3, sellerDetails: 4, eventRequests: 5,
    payouts: 6, payoutReports: 7, relevance: 8, removals: 9 });
  expect(total).toBe(45);
});

it("a missing KPI field gives no count, and the total sums only the present ones", () => {
  kpis.value = { ...base, openFlags: 2 };
  const { byKey, total } = renderHook(() => useAdminQueueCounts()).result.current;
  expect(byKey.sellerDetails).toBeNull();
  expect(byKey.payouts).toBeNull();
  expect(byKey.relevance).toBeNull();
  expect(byKey.removals).toBeNull();
  expect(total).toBe(3); // 1 dispute + 2 flags
});

it("gives no counts and no total while the KPIs load", () => {
  const { byKey, total } = renderHook(() => useAdminQueueCounts()).result.current;
  expect(Object.values(byKey).every((c) => c === null)).toBe(true);
  expect(total).toBeNull();
  expect(renderHook(() => useAdminAttentionTarget()).result.current).toBeNull();
});

it("the dispute dot follows disputesNeedingAdmin, not openDisputes", () => {
  kpis.value = { ...base, disputesNeedingAdmin: 4, disputesOverdue: 1 };
  expect(renderHook(() => useAdminAttentionTarget()).result.current?.count).toBe(4);
  expect(renderHook(() => useAdminQueueCounts()).result.current.byKey.disputes).toBe(4);
});

it("the dispute count falls back to openDisputes while the backend lacks the new field", () => {
  kpis.value = base;
  expect(renderHook(() => useAdminAttentionTarget()).result.current?.count).toBe(1);
  expect(renderHook(() => useAdminQueueCounts()).result.current.byKey.disputes).toBe(1);
});

it("the attention dot goes to the first queue in the order with work", () => {
  kpis.value = { ...full, disputesNeedingAdmin: 0, openFlags: 0, pendingVerifications: 0 };
  expect(renderHook(() => useAdminAttentionTarget()).result.current).toMatchObject({
    label: "Seller details", count: 4, href: "/admin/verifications#seller-details",
  });
});

it("has no Flags queue when the flags gate is off", async () => {
  vi.resetModules();
  vi.doMock("@/lib/constants", async (original) => ({ ...await original<object>(), ADMIN_FLAGS_ENABLED: false }));
  const gated = await import("./admin-queues");
  expect(gated.ADMIN_QUEUES.map((q) => q.key)).not.toContain("flags");
  vi.doUnmock("@/lib/constants");
});
