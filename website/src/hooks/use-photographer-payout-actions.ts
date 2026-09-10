"use client";

import { useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError } from "@/lib/api";
import { requestPhotographerPayout, withdrawPhotographerPayout } from "@/lib/api-photographer-earnings";
import type { PhotographerPayout as PhotographerPayoutRecord } from "@/lib/photographer-mock";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

export const photographerPayoutKeys = {
  list: (userId: string, role: "PHOTOGRAPHER", version: number) => [
    "photographer", "payouts", userId, role, version,
  ] as const,
  balance: (userId: string, role: "PHOTOGRAPHER", version: number) => [
    "photographer", "payouts", "balance", userId, role, version,
  ] as const,
};

export function usePhotographerPayoutActions() {
  const qc = useQueryClient();
  return useMemo(() => {
    const settle = async (
      label: string,
      operation: () => Promise<PhotographerPayoutRecord | void>,
    ): Promise<boolean> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || session.user?.role !== "PHOTOGRAPHER") {
        return false;
      }
      const owner = session.user.id;
      const role = session.user.role;
      const version = session.sessionVersion;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated &&
          current.user?.id === owner &&
          current.user?.role === role &&
          current.sessionVersion === version;
      };

      try {
        await operation();
      } catch (error) {
        if (!isCurrent()) return false;
        const apiMessage = error instanceof ApiError ? error.errors[0]?.message : undefined;
        const fallback = `Couldn't ${label} — please try again.`;
        const message = apiMessage ?? fallback;
        useToastStore.getState().showToast({ kind: "error", message });
        return false;
      }

      if (!isCurrent()) return false;

      await Promise.all([
        qc.cancelQueries({ queryKey: photographerPayoutKeys.list(owner, role, version) }),
        qc.cancelQueries({ queryKey: photographerPayoutKeys.balance(owner, role, version) }),
      ]);

      if (!isCurrent()) return false;

      await Promise.all([
        qc.invalidateQueries({ queryKey: photographerPayoutKeys.list(owner, role, version) }),
        qc.invalidateQueries({ queryKey: photographerPayoutKeys.balance(owner, role, version) }),
      ]);

      return isCurrent();
    };

    return {
      request: () => settle("request payout", () => requestPhotographerPayout()),
      withdraw: (cycleId: string) => settle("withdraw payout", () => withdrawPhotographerPayout(cycleId)),
    };
  }, [qc]);
}
