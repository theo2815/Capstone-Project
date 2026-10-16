"use client";

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getAccessToken } from "@/lib/auth";
import { buildWsUrl } from "@/lib/ws-url";
import { useAuthStore } from "@/store/auth-store";
import { inboxKeys, type InboxRole } from "@/hooks/use-inbox-messages";
import { removalKeys } from "@/hooks/use-photo-removal";

export function useInboxNotificationsWs(role: InboxRole, enabled: boolean): boolean {
  const qc = useQueryClient();
  const owner = useAuthStore((s) => s.user?.id ?? null);
  const session = useAuthStore((s) => s.sessionVersion);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!enabled || !owner) return;
    let cancelled = false;
    let socket: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    const queryKey = inboxKeys.messages(role, owner, session);
    const isCurrent = () => {
      const current = useAuthStore.getState();
      return !cancelled && current.isAuthenticated && current.user?.id === owner &&
        current.user.role === role && current.sessionVersion === session;
    };
    const refresh = async (removalChanged = true) => {
      if (!isCurrent()) return;
      // Also cancel an initial read: an event may arrive before it resolves.
      await qc.cancelQueries({ queryKey, exact: true });
      if (isCurrent()) await qc.invalidateQueries({ queryKey, exact: true });
      if (removalChanged && role === "PHOTOGRAPHER" && isCurrent()) {
        const removalKey = removalKeys.photographer([owner, role, session]);
        await qc.cancelQueries({ queryKey: removalKey });
        if (isCurrent()) await qc.invalidateQueries({ queryKey: removalKey });
      }
    };
    function connect() {
      if (!isCurrent()) return;
      const token = getAccessToken();
      if (!token) return;
      const ws = new WebSocket(buildWsUrl(`/ws/me/${role.toLowerCase()}/notifications`), [token]);
      socket = ws;
      ws.onopen = () => {
        if (!isCurrent() || socket !== ws) return;
        setConnected(true);
        attempts = 0;
        void refresh();
      };
      ws.onmessage = (event) => {
        if (!isCurrent() || socket !== ws) return;
        try {
          const frame = JSON.parse(event.data);
          if (frame.type === "message.created" && typeof frame.message?.id === "string") {
            void refresh(typeof frame.message.photoRemovalRequestId === "string");
          }
        } catch { /* Ignore malformed frames. REST remains authoritative. */ }
      };
      ws.onclose = () => {
        if (!isCurrent() || socket !== ws) return;
        setConnected(false);
        socket = null;
        timer = setTimeout(connect, Math.min(30_000, 1000 * 2 ** Math.min(attempts++, 5)));
      };
      ws.onerror = () => { /* onclose schedules reconnect. */ };
    }
    const onFocus = () => { if (!document.hidden) void refresh(); };
    window.addEventListener("focus", onFocus);
    connect();
    return () => {
      cancelled = true;
      clearTimeout(timer);
      window.removeEventListener("focus", onFocus);
      socket?.close();
      setConnected(false);
    };
  }, [enabled, owner, session, role, qc]);
  return connected;
}
