"use client";

import { useEffect, useRef } from "react";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl } from "@/lib/ws-url";
import { useAdminUsersServerStore } from "@/lib/admin-users-data";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";

// WebSocket push for the admin queue. Subscribes to /ws/admin/notifications
// (handshake gated on ADMIN role server-side) and reacts to every
// AdminInboxEvent emitted by the BE:
//
//   - verification_submitted → invalidate useAdminUsersServerStore so the
//     /admin/photographers and /admin/inbox queues pick up the new row on
//     next read. Active subscribers already re-read via the useEffect
//     keyed on `fetchedAt`, so this lands without page interaction.
//
//   - dispute_filed         → no-op for now. The admin dispute store is
//   - payout_report_filed     mock-fed in v1; Phase G migrates them onto
//                             real BE fetchers and at that point this hook
//                             becomes the invalidation dispatch site.
//                             Logged at info so the BE push is visible in
//                             DevTools even before the store is wired.
//
// Reconnect policy mirrors usePhotographerNotificationsWs — indefinite
// retry with 1s → 30s backoff.

const MAX_BACKOFF_MS = 30_000;

interface AdminInboxFrame {
  type: "verification_submitted" | "dispute_filed" | "payout_report_filed";
  entityId: string;
  actorId: string;
  occurredAt: string;
}

export function useAdminNotificationsWs(enabled: boolean): void {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptsRef = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    function clearReconnectTimer() {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    }

    function connect() {
      if (cancelled) return;
      const url = buildWsUrl("/ws/admin/notifications");
      const token = getAccessToken();
      if (!token) return;
      const ws = new WebSocket(url, [token]);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        attemptsRef.current = 0;
        // Publish status so useAdminQueueRealtime can drop to the slow
        // 5-min heartbeat — the push channel is doing the heavy lifting.
        useAdminWsStatusStore.getState().setStatus("healthy");
        // Reconnect grace — flush the user cache so any verification
        // submission that landed during the down window surfaces on the
        // next read by the queue page.
        useAdminUsersServerStore.getState().invalidate();
        void useAdminUsersServerStore.getState().refetch();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as AdminInboxFrame;
          switch (data.type) {
            case "verification_submitted":
              useAdminUsersServerStore.getState().invalidate();
              void useAdminUsersServerStore.getState().refetch();
              break;
            case "dispute_filed":
            case "payout_report_filed":
              // TODO(Phase G): invalidate admin-dispute-store /
              // admin-payout-report-store once they migrate off mocks.
              console.info(
                `[admin/ws] ${data.type} entity=${data.entityId} actor=${data.actorId}`,
              );
              break;
            default:
              // Unknown type — log and drop.
              console.warn("[admin/ws] unknown frame", data);
          }
        } catch {
          // Silently drop malformed frames.
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
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
  }, [enabled]);
}
