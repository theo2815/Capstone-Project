"use client";

import { captureAuthSession } from "@/lib/auth-session";
import { useAuthStore } from "@/store/auth-store";

import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";
import { adminUserKeys } from "@/hooks/use-admin-users";
import { removalKeys } from "@/hooks/use-photo-removal";

// Adaptive polling fallback for the admin queues. Pairs with
// useAdminNotificationsWs — when the push channel is up we run a slow
// 5-min heartbeat (catches any pushes the client missed during a brief
// disconnect that ws.onclose somehow didn't surface); when it's down we
// drop to 30 s so the admin queue doesn't silently go stale.
//
// invalidateQueries is observer-aware: keys with no mounted observer no-op
// (no fetch fires), so this is essentially free on pages that don't read
// the admin queues. All migrated queues share observer-aware invalidation.
//
// Closes A-2. See [[website/decisions#2026-05-17...]] for the original
// claim this hook fulfills.

const HEALTHY_INTERVAL_MS = 5 * 60_000;
const DEGRADED_INTERVAL_MS = 30_000;

export function useAdminQueueRealtime(enabled: boolean): void {
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);
  const sessionVersion = useAuthStore((s) => s.sessionVersion);
  const status = useAdminWsStatusStore((s) => s.status);

  useEffect(() => {
    if (!enabled || user?.role !== "ADMIN" || !useAuthStore.getState().isAuthenticated) return;
    const isCurrent = captureAuthSession();
    const intervalMs =
      status === "healthy" ? HEALTHY_INTERVAL_MS : DEGRADED_INTERVAL_MS;

    const tick = () => {
      // A backgrounded admin tab has nobody looking at it — skip the pull
      // The next visible tick catches up.
      if (document.hidden || !isCurrent()) return;
      queryClient.invalidateQueries({ queryKey: ["admin", "disputes"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "payouts"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "events"] });
      queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
      void queryClient.invalidateQueries({ queryKey: adminUserKeys.all });
      void queryClient.invalidateQueries({ queryKey: removalKeys.admin([user.id, user.role, sessionVersion]) });
    };

    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [enabled, status, queryClient, user?.id, user?.role, sessionVersion]);
}
