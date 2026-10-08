"use client";

import { captureAuthSession } from "@/lib/auth-session";
import { useAuthStore } from "@/store/auth-store";

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl } from "@/lib/ws-url";
import { adminUserKeys } from "@/hooks/use-admin-users";
import { removalKeys } from "@/hooks/use-photo-removal";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";

// WebSocket push for the admin queue. Subscribes to /ws/admin/notifications
// (handshake gated on ADMIN role server-side) and reacts to every
// AdminInboxEvent emitted by the BE:
//
//   - verification_submitted → invalidate user queries and KPI counts.
//
//   - event_submitted       → invalidate ["admin","events"] + ["admin","kpis"]
//   - event_change_requested  so the V46 event requests queue and its counts
//                             refresh without a reload.
//
//   - dispute_filed / payout_report_filed → invalidate the matching Query
//     lists and KPI counts so incoming work is visible immediately.
//
// Reconnect policy mirrors usePhotographerNotificationsWs — indefinite
// retry with 1s → 30s backoff.

const MAX_BACKOFF_MS = 30_000;

interface AdminInboxFrame {
  type:
    | "verification_submitted"
    | "dispute_filed"
    | "payout_report_filed"
    | "event_submitted"
    | "event_change_requested"
    | "photo_removal_changed";
  entityId: string;
  actorId: string;
  occurredAt: string;
}

export function useAdminNotificationsWs(enabled: boolean): void {
  const queryClient = useQueryClient();
  const user = useAuthStore((s) => s.user);
  const sessionVersion = useAuthStore((s) => s.sessionVersion);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptsRef = useRef(0);

  useEffect(() => {
    if (!enabled || user?.role !== "ADMIN" || !useAuthStore.getState().isAuthenticated) return;
    const isCurrent = captureAuthSession();
    const removalKey = removalKeys.admin([user.id, user.role, sessionVersion]);
    let cancelled = false;

    function clearReconnectTimer() {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    }

    function connect() {
      if (cancelled || !isCurrent()) return;
      const url = buildWsUrl("/ws/admin/notifications");
      const token = getAccessToken();
      if (!token) return;
      const ws = new WebSocket(url, [token]);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled || !isCurrent()) return;
        attemptsRef.current = 0;
        // Publish status so useAdminQueueRealtime can drop to the slow
        // 5-min heartbeat — the push channel is doing the heavy lifting.
        useAdminWsStatusStore.getState().setStatus("healthy");
        // Reconnect grace — flush the user cache so any verification
        // submission that landed during the down window surfaces on the
        // next read by the queue page.
        void queryClient.invalidateQueries({ queryKey: adminUserKeys.all });
        void queryClient.invalidateQueries({ queryKey: ["admin", "disputes"] });
        void queryClient.invalidateQueries({ queryKey: ["admin", "payouts"] });
        void queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
        void queryClient.invalidateQueries({ queryKey: removalKey });
      };

      ws.onmessage = (event) => {
        if (cancelled || !isCurrent()) return;
        try {
          const data = JSON.parse(event.data) as AdminInboxFrame;
          switch (data.type) {
            case "photo_removal_changed":
              void queryClient.invalidateQueries({ queryKey: removalKey });
              void queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
              break;
            case "verification_submitted":
              void queryClient.invalidateQueries({ queryKey: adminUserKeys.all });
              void queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
              break;
            case "event_submitted":
            case "event_change_requested":
              void queryClient.invalidateQueries({ queryKey: ["admin", "events"] });
              void queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
              break;
            case "dispute_filed":
              void queryClient.invalidateQueries({ queryKey: ["admin", "disputes"] });
              void queryClient.invalidateQueries({ queryKey: ["admin", "kpis"] });
              break;
            case "payout_report_filed":
              void queryClient.invalidateQueries({ queryKey: ["admin", "payouts", "reports"] });
              break;
            default:
              // Unknown type — log and drop (dev only — entity/actor ids
              // don't belong in a production console).
              if (process.env.NODE_ENV !== "production") {
                console.warn("[admin/ws] unknown frame", data);
              }
          }
        } catch {
          // Silently drop malformed frames.
        }
      };

      ws.onclose = () => {
        if (cancelled || !isCurrent()) return;
        wsRef.current = null;
        // Connection gone — flip to the aggressive 30-s polling cadence
        // until ws.onopen brings us back. Reconnect attempts continue in
        // parallel via the backoff timer below.
        useAdminWsStatusStore.getState().setStatus("degraded");
        attemptsRef.current += 1;
        const delay = Math.min(
          MAX_BACKOFF_MS,
          1000 * 2 ** Math.min(attemptsRef.current, 5),
        );
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // onclose follows.
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      wsRef.current?.close();
      wsRef.current = null;
      // Teardown (signed out, role change, unmount) — force degraded so a
      // subsequent re-mount starts with the aggressive interval until the
      // new WS connects.
      useAdminWsStatusStore.getState().setStatus("degraded");
    };
  }, [enabled, queryClient, user?.id, user?.role, sessionVersion]);
}
