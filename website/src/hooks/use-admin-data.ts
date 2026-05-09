"use client";

import { useQuery } from "@tanstack/react-query";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  fetchAdminKpis,
  fetchAdminKpiTrend,
  fetchAdminUsers,
  fetchAdminDisputes,
  fetchAdminPayouts,
  fetchAdminPayoutReports,
  fetchAdminEvents,
  fetchAdminSalesKpis,
  fetchAdminSalesByEvent,
  type AdminKpis,
  type AdminTrendPoint,
  type AdminUserStatusFilter,
  type AdminDisputeStatusFilter,
  type AdminSalesEventRow,
  type AdminSalesKpis,
  type AdminSalesRange,
} from "@/lib/api-admin";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import type { Dispute } from "@/lib/admin-disputes";
import type {
  AdminPayoutCycle,
  AdminPayoutStatus,
} from "@/lib/admin-payouts";
import type {
  PayoutReport,
  PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { ListEvent } from "@/app/events/events-browser";

// Hybrid hooks for admin reads. Each hook returns canonical server data
// when `BACKEND_LIVE=true`; in mock mode they return `null` so callers
// can fall back to their existing store-derived computation. Components
// pattern-match on `live ?? derived`.
//
// React Query keys: ["admin", <domain>, ...]
// Stale time: 60s for queue listings, 30s for KPIs, 60s for sales.

const KPI_STALE_MS = 30_000;
const LIST_STALE_MS = 60_000;
const SALES_STALE_MS = 60_000;

// ───────────────────────────────────────────── KPIs

export function useAdminKpis(): AdminKpis | null {
  const query = useQuery<AdminKpis | null>({
    queryKey: ["admin", "kpis"],
    queryFn: () => fetchAdminKpis(),
    enabled: BACKEND_LIVE,
    staleTime: KPI_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

export function useAdminKpiTrend(
  days: number = 30,
): AdminTrendPoint[] | null {
  const query = useQuery<AdminTrendPoint[] | null>({
    queryKey: ["admin", "kpis", "trend", days],
    queryFn: () => fetchAdminKpiTrend(days),
    enabled: BACKEND_LIVE,
    staleTime: KPI_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Users / verifications

export interface UseAdminUsersArgs {
  role?: "PHOTOGRAPHER" | "RUNNER";
  status?: AdminUserStatusFilter;
  q?: string;
}

export function useAdminUsers(
  args: UseAdminUsersArgs = {},
): AdminUserRow[] | null {
  const query = useQuery<AdminUserRow[]>({
    queryKey: ["admin", "users", args],
    queryFn: () => fetchAdminUsers(args),
    enabled: BACKEND_LIVE,
    staleTime: LIST_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Disputes

export function useAdminDisputes(args: {
  status?: AdminDisputeStatusFilter;
  q?: string;
} = {}): Dispute[] | null {
  const query = useQuery<Dispute[]>({
    queryKey: ["admin", "disputes", args],
    queryFn: () => fetchAdminDisputes(args),
    enabled: BACKEND_LIVE,
    staleTime: LIST_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Payouts

export function useAdminPayouts(args: {
  status?: AdminPayoutStatus;
  q?: string;
} = {}): AdminPayoutCycle[] | null {
  const query = useQuery<AdminPayoutCycle[]>({
    queryKey: ["admin", "payouts", args],
    queryFn: () => fetchAdminPayouts(args),
    enabled: BACKEND_LIVE,
    staleTime: LIST_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Payout reports

export function useAdminPayoutReports(args: {
  status?: PayoutReportStatus;
} = {}): PayoutReport[] | null {
  const query = useQuery<PayoutReport[]>({
    queryKey: ["admin", "payouts", "reports", args],
    queryFn: () => fetchAdminPayoutReports(args),
    enabled: BACKEND_LIVE,
    staleTime: LIST_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Events catalog

export function useAdminEvents(args: {
  state?: ListEvent["state"];
} = {}): ListEvent[] | null {
  const query = useQuery<ListEvent[]>({
    queryKey: ["admin", "events", args],
    queryFn: () => fetchAdminEvents(args),
    enabled: BACKEND_LIVE,
    staleTime: LIST_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Sales

export function useAdminSalesKpisLive(
  range: AdminSalesRange = "ytd",
): AdminSalesKpis | null {
  const query = useQuery<AdminSalesKpis | null>({
    queryKey: ["admin", "sales", "kpis", range],
    queryFn: () => fetchAdminSalesKpis(range),
    enabled: BACKEND_LIVE,
    staleTime: SALES_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

export function useAdminSalesByEventLive(args: {
  order?: "gmv" | "refunds";
} = {}): AdminSalesEventRow[] | null {
  const query = useQuery<AdminSalesEventRow[] | null>({
    queryKey: ["admin", "sales", "by-event", args],
    queryFn: () => fetchAdminSalesByEvent(args),
    enabled: BACKEND_LIVE,
    staleTime: SALES_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}
