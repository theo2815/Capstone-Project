"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { buildWsUrl } from "@/lib/ws-url";
import { getAccessToken } from "@/lib/auth";
import type { Photo } from "@/lib/api-photos";

interface PhotoPublishedMessage {
  type: "photo.published";
  photo: Photo;
}

interface UseEventLivePhotosArgs {
  slug: string;
  eventId: string;
  // Caller passes `event.state === "live"`; the hook only opens a socket
  // when this is true.
  enabled: boolean;
}

interface UseEventLivePhotosResult {
  isConnected: boolean;
  newCount: number;
  reconnectFailed: boolean;
  refresh: () => void;
}

const MAX_RECONNECT_ATTEMPTS = 3;
const MAX_BACKOFF_MS = 30_000;

// Q-002: WebSocket push for live photo arrival on `/events/[slug]?browse=1`.
// Mounts only on `live` events; closes on unmount. After 3 failed reconnects
// the caller surfaces a manual "Refresh" button.
export function useEventLivePhotos(
  args: UseEventLivePhotosArgs,
): UseEventLivePhotosResult {
  const qc = useQueryClient();
  const [isConnected, setIsConnected] = useState(false);
  const [newCount, setNewCount] = useState(0);
  const [reconnectFailed, setReconnectFailed] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const failedAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const seenPhotoIdsRef = useRef(new Set<string>());

  useEffect(() => {
    if (!args.enabled) return;
    let cancelled = false;

    function clearReconnectTimer() {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    }

    function connect() {
      if (cancelled) return;
      // Bearer JWT carried via Sec-WebSocket-Protocol per Q-002. Backend
      // reads the first protocol value as the access token.
      const token = getAccessToken();
      // `EventPhotoHandshakeInterceptor.beforeHandshake` returns false when no
      // token is present, so a tokenless socket can only fail the upgrade. The
      // browser reports that as a 1006 close with no `onopen`, which walks the
      // retry ladder and lands every signed-out visitor on a live event at the
      // false "Connection lost · Refresh" banner. Don't open what will be
      // refused — guests simply get no live push (tracked in backend/tasks.md).
      if (!token) return;
      // BE registers the handler at /ws/events/*/photos — NOT under the REST
      // /api/v1 prefix. buildWsUrl strips API_BASE_URL's path component before
      // appending the WS path; matches use-admin-notifications-ws + use-
      // photographer-notifications-ws.
      const url = buildWsUrl(
        `/ws/events/${encodeURIComponent(args.eventId)}/photos`,
      );
      const ws = new WebSocket(url, [token]);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setIsConnected(true);
        setReconnectFailed(false);
        failedAttemptsRef.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as PhotoPublishedMessage;
          if (data.type !== "photo.published") return;
          // Keep the current snapshot stable. The badge lets the visitor pull
          // a fresh, diversity-ranked first page without live arrivals jumping
          // ahead of photographers who have not been represented yet.
          if (seenPhotoIdsRef.current.has(data.photo.id)) return;
          seenPhotoIdsRef.current.add(data.photo.id);
          setNewCount((c) => c + 1);
        } catch {
          // Silently drop malformed pushes.
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        setIsConnected(false);
        wsRef.current = null;
        failedAttemptsRef.current += 1;
        if (failedAttemptsRef.current > MAX_RECONNECT_ATTEMPTS) {
          setReconnectFailed(true);
          return;
        }
        const delay = Math.min(
          MAX_BACKOFF_MS,
          1000 * 2 ** failedAttemptsRef.current,
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
    };
  }, [args.enabled, args.slug, args.eventId, qc]);

  return {
    isConnected,
    newCount,
    reconnectFailed,
    refresh: () => {
      seenPhotoIdsRef.current.clear();
      qc.invalidateQueries({ queryKey: ["events", args.slug, "photos"] });
      setReconnectFailed(false);
      failedAttemptsRef.current = 0;
      setNewCount(0);
    },
  };
}
