"use client";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminDisputes, fetchAdminDispute, resolveDispute, denyDispute, escalateDispute,
  type AdminDisputeListArgs, type ResolveDisputeArgs,
} from "@/lib/api-admin";
import type { Dispute } from "@/lib/admin-disputes";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

export const EMPTY_DISPUTES: Dispute[] = [];
export const adminDisputeKeys = {
  all: ["admin", "disputes"] as const,
  owner: (userId: string | null) => ["admin", "disputes", userId] as const,
  lists: (userId: string | null) => [...adminDisputeKeys.owner(userId), "list"] as const,
  list: (userId: string | null, args: AdminDisputeListArgs = {}) =>
    [...adminDisputeKeys.lists(userId), args] as const,
  detail: (userId: string | null, id: string) =>
    [...adminDisputeKeys.owner(userId), "detail", id] as const,
};

export function useAdminDisputes(args: AdminDisputeListArgs = {}) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminDisputeKeys.list(user?.id ?? null, args),
    queryFn: () => fetchAdminDisputes(args),
    enabled: user?.role === "ADMIN",
    staleTime: 60_000,
  });
}

export function useAdminDispute(id: string) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminDisputeKeys.detail(user?.id ?? null, id),
    queryFn: ({ signal }) => fetchAdminDispute(id, signal),
    enabled: user?.role === "ADMIN" && !!id,
    staleTime: 60_000,
  });
}

// The API returns the complete record, including its persisted activity.
// Replace cached rows with that record; never reconstruct a decision locally.
export function useDisputeActions() {
  const qc = useQueryClient();
  return useMemo(() => {
    const settle = async (label: string, operation: () => Promise<Dispute>): Promise<Dispute | null> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || session.user?.role !== "ADMIN") return null;
      const userId = session.user.id;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated && current.user?.id === userId && current.sessionVersion === session.sessionVersion;
      };
      let dispute: Dispute;
      try {
        dispute = await operation();
      } catch (error) {
        if (isCurrent()) {
          console.error(`[admin/disputes] ${label} failed`, error);
          useToastStore.getState().showToast({
            kind: "error", message: `Couldn't ${label} dispute — please try again.`,
          });
        }
        return null;
      }
      if (!isCurrent()) return null;
      // An older read must not overwrite the confirmed mutation response.
      await qc.cancelQueries({ queryKey: adminDisputeKeys.owner(userId) });
      if (!isCurrent()) return null;
      for (const [key] of qc.getQueriesData<Dispute[]>({ queryKey: adminDisputeKeys.lists(userId) })) {
        const args = key.at(-1) as AdminDisputeListArgs;
        qc.setQueryData<Dispute[]>(key, (rows) => rows?.flatMap((row) => {
          if (row.id !== dispute.id) return [row];
          return args.status && args.status !== dispute.status ? [] : [dispute];
        }));
      }
      qc.setQueryData(adminDisputeKeys.detail(userId, dispute.id), dispute);
      // Refetch lists for filter membership/pagination and related financial counts.
      await Promise.all([
        qc.invalidateQueries({ queryKey: adminDisputeKeys.lists(userId) }),
        qc.invalidateQueries({ queryKey: ["admin", "kpis"] }),
        qc.invalidateQueries({ queryKey: ["admin", "sales"] }),
      ]);
      return isCurrent() ? dispute : null;
    };
    return {
      resolve: (id: string, args: ResolveDisputeArgs) => settle("resolve", () => resolveDispute(id, args)),
      deny: async (id: string, reason: string | null) => !!(await settle("deny", () => denyDispute(id, reason))),
      escalate: async (id: string, reason: string | null) => !!(await settle("escalate", () => escalateDispute(id, reason))),
    };
  }, [qc]);
}

const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

export function useOpenDisputesCount(): number {
  const serverDisputes = useAdminDisputes().data ?? EMPTY_DISPUTES;
  return useMemo(
    () =>
      serverDisputes.filter(
        (d) => d.status === "open",
      ).length,
    [serverDisputes],
  );
}

export function useRefundedThisWeekTotal(): number {
  const serverDisputes = useAdminDisputes().data ?? EMPTY_DISPUTES;
  return useMemo(() => {
    const cutoff = Date.now() - SEVEN_DAYS_MS;
    return serverDisputes
      .filter((d) => {
        if (d.status !== "resolved" || d.refundAmount === null) return false;
        const t = new Date(d.resolvedAt ?? "").getTime();
        return Number.isFinite(t) && t >= cutoff;
      })
      .reduce((acc, d) => acc + (d.refundAmount ?? 0), 0);
  }, [serverDisputes]);
}

