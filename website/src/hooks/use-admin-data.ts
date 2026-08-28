"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchAdminKpis,
  fetchAdminKpiTrend,
  fetchAdminDisputes,
  fetchAdminPayouts,
  fetchAdminPayoutReports,
  fetchAdminEvents,
  fetchAdminSalesKpis,
  fetchAdminSalesByEvent,
  type AdminKpis,
  type AdminTrendPoint,
  type AdminDisputeStatusFilter,
  type AdminSalesEventRow,
  type AdminSalesKpis,
  type AdminSalesRange,
} from "@/lib/api-admin";
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

// React Query hooks for admin reads.
// Keys: ["admin", <domain>, ...]
// Stale time: 60s for queue listings, 30s for KPIs, 60s for sales.

const KPI_STALE_MS = 30_000;
const LIST_STALE_MS = 60_000;
const SALES_STALE_MS = 60_000;

// Stable empty fallbacks for the `useX() ?? EMPTY_*` pattern at call sites.
// A literal `?? []` mints a fresh array identity on every render while the
// query loads, which defeats every downstream useMemo keyed on it (the whole
// derive chain recomputes per render). Same trick as the upload page's
// EMPTY_SEED.
export const EMPTY_DISPUTES: Dispute[] = [];
export const EMPTY_PAYOUTS: AdminPayoutCycle[] = [];
export const EMPTY_REPORTS: PayoutReport[] = [];

// ───────────────────────────────────────────── KPIs

export function useAdminKpis(): AdminKpis | null {
  const query = useQuery<AdminKpis>({
    queryKey: ["admin", "kpis"],
    queryFn: () => fetchAdminKpis(),
    staleTime: KPI_STALE_MS,
  });
  return query.data ?? null;
}

export function useAdminKpiTrend(
  days: number = 30,
): AdminTrendPoint[] | null {
  const query = useQuery<AdminTrendPoint[]>({
    queryKey: ["admin", "kpis", "trend", days],
    queryFn: () => fetchAdminKpiTrend(days),
    staleTime: KPI_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Disputes

export function useAdminDisputes(args: {
  status?: AdminDisputeStatusFilter;
  q?: string;
} = {}): Dispute[] | null {
  const query = useQuery<Dispute[]>({
    queryKey: ["admin", "disputes", args],
    queryFn: () => fetchAdminDisputes(args),
    staleTime: LIST_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Payouts

export function useAdminPayouts(args: {
  status?: AdminPayoutStatus;
  q?: string;
} = {}): AdminPayoutCycle[] | null {
  const query = useQuery<AdminPayoutCycle[]>({
    queryKey: ["admin", "payouts", args],
    queryFn: () => fetchAdminPayouts(args),
    staleTime: LIST_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Payout reports

export function useAdminPayoutReports(args: {
  status?: PayoutReportStatus;
} = {}): PayoutReport[] | null {
  const query = useQuery<PayoutReport[]>({
    queryKey: ["admin", "payouts", "reports", args],
    queryFn: () => fetchAdminPayoutReports(args),
    staleTime: LIST_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Events catalog

export function useAdminEvents(args: {
  state?: ListEvent["state"];
} = {}): ListEvent[] | null {
  const query = useQuery<ListEvent[]>({
    queryKey: ["admin", "events", args],
    queryFn: () => fetchAdminEvents(args),
    staleTime: LIST_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Sales

export function useAdminSalesKpisLive(
  range: AdminSalesRange = "ytd",
): AdminSalesKpis | null {
  const query = useQuery<AdminSalesKpis>({
    queryKey: ["admin", "sales", "kpis", range],
    queryFn: () => fetchAdminSalesKpis(range),
    staleTime: SALES_STALE_MS,
  });
  return query.data ?? null;
}

export function useAdminSalesByEventLive(args: {
  order?: "gmv" | "refunds";
} = {}): AdminSalesEventRow[] | null {
  const query = useQuery<AdminSalesEventRow[]>({
    queryKey: ["admin", "sales", "by-event", args],
    queryFn: () => fetchAdminSalesByEvent(args),
    staleTime: SALES_STALE_MS,
  });
  return query.data ?? null;
}
