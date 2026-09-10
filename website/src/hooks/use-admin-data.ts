"use client";

import { usePrivateQuerySession } from "@/hooks/use-private-query-session";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminKpis,
  fetchAdminKpiTrend,
  fetchAdminFlags,
  hideAdminFlag,
  dismissAdminFlag,
  escalateAdminFlag,
  fetchAdminSalesKpis,
  fetchAdminSalesByEvent,
  type AdminKpis,
  type AdminTrendPoint,
  type AdminFlagListArgs,
  type AdminSalesEventRow,
  type AdminSalesKpis,
  type AdminSalesRange,
} from "@/lib/api-admin";
import type { Flag } from "@/lib/admin-flags";
import type { PaginatedResponse } from "@/types/api";
import { ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { captureAuthSession } from "@/lib/auth-session";
import { useAuthStore } from "@/store/auth-store";

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
export const EMPTY_FLAGS: Flag[] = [];

// ───────────────────────────────────────────── KPIs

export function useAdminKpis(): AdminKpis | null {
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery<AdminKpis>({
    queryKey: ["admin", "kpis", ...session.key],
    queryFn: ({ signal }) => fetchAdminKpis(signal),
    enabled: session.enabled,
    staleTime: KPI_STALE_MS,
  });
  return query.data ?? null;
}

export function useAdminKpiTrend(
  days: number = 30,
): AdminTrendPoint[] | null {
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery<AdminTrendPoint[]>({
    queryKey: ["admin", "kpis", "trend", days, ...session.key],
    queryFn: ({ signal }) => fetchAdminKpiTrend(days, signal),
    enabled: session.enabled,
    staleTime: KPI_STALE_MS,
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
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery<PaginatedResponse<Flag>>({
    queryKey: ["admin", "flags", args, ...session.key],
    queryFn: ({ signal }) => fetchAdminFlags(args, signal),
    staleTime: LIST_STALE_MS,
    enabled: session.enabled && ADMIN_FLAGS_ENABLED,
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
    const settle = async (operation: () => Promise<unknown>): Promise<boolean> => {
      const auth = useAuthStore.getState();
      if (!auth.isAuthenticated || auth.user?.role !== "ADMIN") return false;
      const isCurrent = captureAuthSession();
      try {
        await operation();
      } catch (error) {
        if (!isCurrent()) return false;
        throw error;
      }
      if (!isCurrent()) return false;
      await Promise.all([
        qc.invalidateQueries({ queryKey: ["admin", "flags"] }),
        qc.invalidateQueries({ queryKey: ["admin", "kpis"] }),
      ]);
      return isCurrent();
    };
    return {
      hide: (flagId: string, reason: string | null) =>
        settle(() => hideAdminFlag(flagId, reason)),
      dismiss: (flagId: string, reason: string | null = null) =>
        settle(() => dismissAdminFlag(flagId, reason)),
      escalate: (flagId: string, note: string | null) =>
        settle(() => escalateAdminFlag(flagId, note)),
    };
  }, [qc]);
}

// ───────────────────────────────────────────── Sales

export function useAdminSalesKpisLive(
  range: AdminSalesRange = "ytd",
): AdminSalesKpis | null {
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery<AdminSalesKpis>({
    queryKey: ["admin", "sales", "kpis", range, ...session.key],
    queryFn: ({ signal }) => fetchAdminSalesKpis(range, signal),
    enabled: session.enabled,
    staleTime: SALES_STALE_MS,
  });
  return query.data ?? null;
}

export function useAdminSalesByEventLive(args: {
  order?: "gmv" | "refunds";
} = {}): AdminSalesEventRow[] | null {
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery<AdminSalesEventRow[]>({
    queryKey: ["admin", "sales", "by-event", args, ...session.key],
    queryFn: ({ signal }) => fetchAdminSalesByEvent(args, signal),
    enabled: session.enabled,
    staleTime: SALES_STALE_MS,
  });
  return query.data ?? null;
}
