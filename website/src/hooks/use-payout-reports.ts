"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminPayoutReports, acknowledgePayoutReport, resolvePayoutReport,
  type AdminPayoutReportListArgs,
} from "@/lib/api-admin";
import {
  fetchPhotographerPayoutReports, submitPayoutReport,
  type PhotographerPayoutReportsArgs, type SubmitPayoutReportArgs,
} from "@/lib/api-photographer-earnings";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { useAdminMutation } from "./use-admin-mutation";

export const EMPTY_REPORTS: PayoutReport[] = [];
export const payoutReportKeys = {
  admin: ["admin", "payouts", "reports"] as const,
  photographer: ["photographer", "payouts", "reports"] as const,
  adminList: (userId: string | null, args: AdminPayoutReportListArgs = {}) =>
    [...payoutReportKeys.admin, userId, args] as const,
  photographerList: (userId: string | null, args: PhotographerPayoutReportsArgs = {}) =>
    [...payoutReportKeys.photographer, userId, args] as const,
};

export function useAdminPayoutReports(args: AdminPayoutReportListArgs = {}) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: payoutReportKeys.adminList(user?.id ?? null, args),
    queryFn: () => fetchAdminPayoutReports(args),
    enabled: user?.role === "ADMIN",
    staleTime: 60_000,
  });
}

export function usePhotographerPayoutReports(args: PhotographerPayoutReportsArgs = {}) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: payoutReportKeys.photographerList(user?.id ?? null, args),
    queryFn: () => fetchPhotographerPayoutReports(args),
    enabled: user?.role === "PHOTOGRAPHER",
    staleTime: 60_000,
  });
}

export function usePayoutReportActions() {
  const qc = useQueryClient();
  const mutation = useAdminMutation();
    const settle = async (label: string, operation: () => Promise<PayoutReport>): Promise<boolean> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || !session.user) return false;
      const userId = session.user.id;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated && current.user?.id === userId && current.sessionVersion === session.sessionVersion;
      };
      let report: PayoutReport;
      try {
        report = await operation();
      } catch (error) {
        if (isCurrent()) {
          console.error(`[payout-reports] ${label} failed`, error);
          useToastStore.getState().showToast({
            kind: "error", message: `Couldn't ${label} payout report — please try again.`,
          });
        }
        return false;
      }
      if (!isCurrent()) return false;
      const prefixes = [payoutReportKeys.admin, payoutReportKeys.photographer];
      await Promise.all(prefixes.map((prefix) => qc.cancelQueries({ queryKey: [...prefix, userId] })));
      if (!isCurrent()) return false;
      for (const prefix of prefixes) {
        for (const [key] of qc.getQueriesData<PayoutReport[]>({ queryKey: [...prefix, userId] })) {
          const args = key.at(-1) as PhotographerPayoutReportsArgs;
          const matches = (!args.status || args.status === report.status) &&
            (!args.cycleId || args.cycleId === report.payoutCycleId);
          qc.setQueryData<PayoutReport[]>(key, (rows) => {
            if (!rows) return rows;
            const existing = rows.some((row) => row.id === report.id);
            if (!existing && label === "submit" && matches && prefix === payoutReportKeys.photographer) {
              return [report, ...rows];
            }
            return rows.flatMap((row) => row.id !== report.id ? [row] : matches ? [report] : []);
          });
        }
      }
      const refresh = prefixes.map((queryKey) => qc.invalidateQueries({ queryKey }));
      if (label === "submit") await Promise.all(refresh);
      else await Promise.allSettled(refresh);
      return isCurrent();
    };
    const run = async (id: string, action: string, label: string, operation: () => Promise<PayoutReport>) => {
      const result = await mutation.run({ targets: [{ kind: "report", id }], action, execute: () => settle(label, operation) });
      return result.kind === "completed" && result.value;
    };
    return {
      isPending: (id: string) => mutation.isPending([{ kind: "report", id }]),
      pendingAction: (id: string) => mutation.pendingAction([{ kind: "report", id }]),
      acknowledge: (id: string, reply: string) => run(id, "Acknowledging…", "acknowledge", () => acknowledgePayoutReport(id, reply)),
      resolve: (id: string, note: string) => run(id, "Resolving…", "resolve", () => resolvePayoutReport(id, note)),
      submit: (args: SubmitPayoutReportArgs) => settle("submit", () => submitPayoutReport(args)),
    };
}
