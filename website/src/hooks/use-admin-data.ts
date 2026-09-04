"use client";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminKpis,
  fetchAdminKpiTrend,
  fetchAdminDisputes,
  fetchAdminFlags,
  hideAdminFlag,
  dismissAdminFlag,
  escalateAdminFlag,
  fetchAdminPayouts,
  fetchAdminPayoutReports,
  fetchAdminEvents,
  fetchAdminSalesKpis,
  fetchAdminSalesByEvent,
  type AdminKpis,
  type AdminTrendPoint,
  type AdminDisputeStatusFilter,
  type AdminFlagListArgs,
  type AdminSalesEventRow,
  type AdminSalesKpis,
  type AdminSalesRange,
} from "@/lib/api-admin";
import type { Dispute } from "@/lib/admin-disputes";
import type { Flag } from "@/lib/admin-flags";
import type {
  AdminPayoutCycle,
  AdminPayoutStatus,
} from "@/lib/admin-payouts";
import type {
  PayoutReport,
  PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { AdminEventListArgs, AdminEventRow } from "@/lib/api-admin";
import type { PaginatedResponse } from "@/types/api";
import { ADMIN_FLAGS_ENABLED } from "@/lib/constants";

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
export const EMPTY_FLAGS: Flag[] = [];
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

// ───────────────────────────────────────────── Flags

// Returns the page (items + total) so callers can tell when the 200-row
// cap truncated the list. Gated on ADMIN_FLAGS_ENABLED here, once, so every
// consumer can call it unconditionally (rules-of-hooks) without firing a
// request the backend would 403.
export function useAdminFlags(
  args: AdminFlagListArgs = {},
): PaginatedResponse<Flag> | null {
  const query = useQuery<PaginatedResponse<Flag>>({
    queryKey: ["admin", "flags", args],
    queryFn: () => fetchAdminFlags(args),
    staleTime: LIST_STALE_MS,
    enabled: ADMIN_FLAGS_ENABLED,
  });
  return query.data ?? null;
}

// Flag actions are server-authoritative: no optimistic override, the call
// awaits the backend and then refetches every flags list + the KPI counts.
// Callers catch the rejection and show an error toast — a lost moderation
// action must never look like a success.
export function useFlagActions() {
  const qc = useQueryClient();
  return useMemo(() => {
    const settle = async (p: Promise<unknown>) => {
      await p;
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["admin", "flags"] }),
        qc.invalidateQueries({ queryKey: ["admin", "kpis"] }),
      ]);
    };
    return {
      hide: (flagId: string, reason: string | null) =>
        settle(hideAdminFlag(flagId, reason)),
      dismiss: (flagId: string, reason: string | null = null) =>
        settle(dismissAdminFlag(flagId, reason)),
      escalate: (flagId: string, note: string | null) =>
        settle(escalateAdminFlag(flagId, note)),
    };
  }, [qc]);
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

export function useAdminEvents(
  args: AdminEventListArgs = {},
): AdminEventRow[] | null {
  const query = useQuery<AdminEventRow[]>({
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
