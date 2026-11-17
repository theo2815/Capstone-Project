import { renderHook } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";
import { useAdminAttentionTarget, useAdminQueueCounts } from "./admin-queues";

const kpis = vi.hoisted(() => ({ value: null as AdminKpis | null }));
vi.mock("@/hooks/use-admin-data", () => ({
  useAdminKpis: () => kpis.value,
  useAdminFlags: () => ({ total: 0 }),
}));
vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: () => ({ data: [] }), EMPTY_PAYOUTS: [] }));
vi.mock("@/hooks/use-admin-users", () => ({ usePendingVerificationsCount: () => 0 }));

const base: AdminKpis = {
  pendingVerifications: 0, approvedPhotographers: 0, suspended: 0, liveEvents: 0,
  decisionsThisWeek: 0, openDisputes: 1, openFlags: 0, pendingPayouts: 0, pendingEventRequests: 0,
};

it("the dispute dot follows disputesNeedingAdmin, not openDisputes", () => {
  kpis.value = { ...base, disputesNeedingAdmin: 4, disputesOverdue: 1 };
  expect(renderHook(() => useAdminAttentionTarget()).result.current?.count).toBe(4);
  expect(renderHook(() => useAdminQueueCounts()).result.current.disputes).toBe(4);
});

it("the dispute count falls back to openDisputes while the backend lacks the new field", () => {
  kpis.value = base;
  expect(renderHook(() => useAdminAttentionTarget()).result.current?.count).toBe(1);
  expect(renderHook(() => useAdminQueueCounts()).result.current.disputes).toBe(1);
});

it("the dot lights when only escalated disputes need an admin", () => {
  kpis.value = { ...base, openDisputes: 0, disputesNeedingAdmin: 2, disputesOverdue: 0 };
  expect(renderHook(() => useAdminAttentionTarget()).result.current?.label).toBe("Disputes");
});
