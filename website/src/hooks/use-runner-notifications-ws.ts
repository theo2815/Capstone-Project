"use client";

import { useEffect, useRef } from "react";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl } from "@/lib/ws-url";
import type { RunnerMessage } from "@/lib/runner-messages";
import { useMyRunnerMessagesStore } from "@/lib/me-runner-messages-data";

// WebSocket push for /me/runner/messages. Subscribes to the BE
// notification channel at /ws/me/runner/notifications, dispatches every
// "message.created" frame into useMyRunnerMessagesStore.applyPush(...).
// The BE handler is shared with the photographer channel — same per-user
// registry, different URL just so the FE knows which inbox to listen on.

const MAX_BACKOFF_MS = 30_000;

interface RunnerNotificationFrame {
  type: "message.created";
  message: RunnerMessage;
}

export function useRunnerNotificationsWs(enabled: boolean): void {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptsRef = useRef(0);
  const setWsConnected = useMyRunnerMessagesStore((s) => s.setWsConnected);
  const applyPush = useMyRunnerMessagesStore((s) => s.applyPush);
  const refetch = useMyRunnerMessagesStore((s) => s.refetch);

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
      const url = buildWsUrl("/ws/me/runner/notifications");
      const token = getAccessToken();
      if (!token) return;
      const ws = new WebSocket(url, [token]);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setWsConnected(true);
        attemptsRef.current = 0;
        // Reconnect grace — close any down-window gap with a REST refetch.
        void refetch();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as RunnerNotificationFrame;
          if (data.type !== "message.created") return;
          applyPush(data.message);
        } catch {
          // Drop malformed frames.
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setWsConnected(false);
        wsRef.current = null;
        attemptsRef.current += 1;
        const delay = Math.min(
          MAX_BACKOFF_MS,
          1000 * 2 ** Math.min(attemptsRef.current, 5),
        );
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        // onclose follows; reconnect schedules there.
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearReconnectTimer();
      setWsConnected(false);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled, applyPush, setWsConnected, refetch]);
}
