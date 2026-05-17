"use client";

import { useEffect, useRef } from "react";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl } from "@/lib/ws-url";
import type { PhotographerMessage } from "@/lib/photographer-messages";
import { useMyPhotographerMessagesStore } from "@/lib/me-photographer-messages-data";

// WebSocket push for /me/photographer/messages. Subscribes to the BE
// notification channel, dispatches every "message.created" frame into
// `useMyPhotographerMessagesStore.applyPush(...)`. The store dedupes by id
// so re-fanned frames (multiple sessions per photographer) collapse to one
// row in the bell.
//
// Reconnect policy is more permissive than useEventLivePhotos: the bell is
// always-on across the photographer's entire session, so giving up after
// 3 retries would silently break the badge. We retry indefinitely with
// 1s → 30s exponential backoff and surface the connection state via the
// store's `wsConnected` flag (drives adaptive REST polling).

const MAX_BACKOFF_MS = 30_000;

interface PhotographerNotificationFrame {
  type: "message.created";
  message: PhotographerMessage;
}

export function usePhotographerNotificationsWs(enabled: boolean): void {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptsRef = useRef(0);
  const setWsConnected = useMyPhotographerMessagesStore(
    (s) => s.setWsConnected,
  );
  const applyPush = useMyPhotographerMessagesStore((s) => s.applyPush);
  const refetch = useMyPhotographerMessagesStore((s) => s.refetch);

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
      const url = buildWsUrl("/ws/me/photographer/notifications");
      const token = getAccessToken();
      if (!token) {
        // Signed-out — no point opening a socket the handshake will reject.
        // The next sign-in flips `enabled` (auth-gated by the caller) and
        // the effect re-runs with a valid token.
        return;
      }
      const ws = new WebSocket(url, [token]);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setWsConnected(true);
        attemptsRef.current = 0;
        // Reconnect grace — the WS may have dropped while a push was emitted.
        // A REST refetch on reconnect closes that gap so the inbox doesn't
        // silently miss a message that arrived during the down window.
        void refetch();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as PhotographerNotificationFrame;
          if (data.type !== "message.created") return;
          applyPush(data.message);
        } catch {
          // Silently drop malformed frames — matches useEventLivePhotos.
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
