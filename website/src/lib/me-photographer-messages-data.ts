"use client";

import { useEffect } from "react";
import { create } from "zustand";
import type { PhotographerMessage } from "@/lib/photographer-messages";
import {
  fetchMyPhotographerMessages,
  markAllMyPhotographerMessagesRead,
  markMyPhotographerMessageRead,
  removeMyPhotographerMessage,
} from "@/lib/api-me-photographer-messages";

// Photographer-side inbox cache + actions. Mirrors the admin-users-data
// pattern: in-flight dedup, focus/visibilitychange refetch, plus mutation
// helpers that update the cache after the BE response so the bell badge +
// inbox modal stay in sync without a refetch round-trip.
//
// Why a global store instead of useState in the bell/modal: the modal
// re-mounts every time it opens, and we don't want the unread badge to
// flicker to zero on first paint. The store survives modal mount/unmount.
//
// Adaptive polling: when the WebSocket push channel reports healthy
// (usePhotographerNotificationsWs flips wsConnected), the stale window
// relaxes to 5min — push is the primary freshness signal. When the WS is
// down, the stale window snaps back to 30s so the bell still catches new
// messages within a normal interaction window. Polling never fully stops
// because the WS reconnect itself can race a missed message; the next
// poll on focus/visibility hides that gap.

const STALE_HEALTHY_MS = 5 * 60_000;
const STALE_DEGRADED_MS = 30_000;
// The BE inbox defaults to 100 rows and caps at 200. Load-more bumps the
// requested limit and re-fetches (the store mixes WS-pushed + optimistically
// mutated rows, so a full replace is safer than offset-appending).
const MESSAGES_PAGE = 100;
const MESSAGES_MAX = 200;

interface MessagesState {
  messages: PhotographerMessage[];
  total: number | null;
  limit: number;
  loading: boolean;
  error: string | null;
  fetchedAt: number;
  inFlight: Promise<void> | null;
  wsConnected: boolean;
  refetch: () => Promise<void>;
  loadMore: () => Promise<void>;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
  remove: (id: string) => Promise<void>;
  /** Called by the WS hook on every push frame. Prepends + dedupes by id —
   *  the BE may re-send if the photographer has multiple sessions and the
   *  TX commit fanned out to all of them. Bumps the unread badge if the
   *  pushed message is unread. */
  applyPush: (msg: PhotographerMessage) => void;
  /** Flips when the WS opens/closes. Drives the adaptive STALE_MS window. */
  setWsConnected: (connected: boolean) => void;
  reset: () => void;
}

export const useMyPhotographerMessagesStore = create<MessagesState>(
  (set, get) => ({
    messages: [],
    total: null,
    limit: MESSAGES_PAGE,
    loading: false,
    error: null,
    fetchedAt: 0,
    inFlight: null,
    wsConnected: false,
    refetch: async () => {
      const existing = get().inFlight;
      if (existing) return existing;
      const p = (async () => {
        set({ loading: true, error: null });
        try {
          const { messages, total } = await fetchMyPhotographerMessages(
            get().limit,
          );
          set({
            messages,
            total,
            loading: false,
            fetchedAt: Date.now(),
            inFlight: null,
          });
        } catch (err) {
          set({
            loading: false,
            error:
              err instanceof Error ? err.message : "Failed to load inbox",
            inFlight: null,
          });
        }
      })();
      set({ inFlight: p });
      return p;
    },
    loadMore: async () => {
      const prev = get().limit;
      const next = Math.min(prev + MESSAGES_PAGE, MESSAGES_MAX);
      if (next === prev) return;
      set({ limit: next, fetchedAt: 0 });
      await get().refetch();
      // refetch swallows its own errors — roll the limit back on failure so
      // hasMore stays true and Load-older remains retryable.
      if (get().error) set({ limit: prev });
    },
    markRead: async (id) => {
      // Optimistic flip — the BE PATCH returns the updated row but the
      // photographer just wants the unread badge to drop without latency.
      // Roll back if the BE rejects (shouldn't happen except on 401 / 5xx).
      const prev = get().messages;
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return;
      if (prev[idx].readAt !== null) return;
      const at = new Date().toISOString();
      const next = prev.slice();
      next[idx] = { ...prev[idx], readAt: at };
      set({ messages: next });
      try {
        const updated = await markMyPhotographerMessageRead(id);
        set((s) => {
          const i = s.messages.findIndex((m) => m.id === id);
          if (i === -1) return s;
          const arr = s.messages.slice();
          arr[i] = updated;
          return { messages: arr };
        });
      } catch (err) {
        console.error("[me/messages] markRead failed", err);
        set({ messages: prev });
      }
    },
    markAllRead: async () => {
      const prev = get().messages;
      const at = new Date().toISOString();
      const next = prev.map((m) =>
        m.readAt === null ? { ...m, readAt: at } : m,
      );
      set({ messages: next });
      try {
        await markAllMyPhotographerMessagesRead();
      } catch (err) {
        console.error("[me/messages] markAllRead failed", err);
        set({ messages: prev });
      }
    },
    remove: async (id) => {
      const prev = get().messages;
      const prevTotal = get().total;
      const next = prev.filter((m) => m.id !== id);
      // Keep the header total in step with the optimistic removal, and
      // restore it on rollback — X-Total-Count only refreshes on refetch.
      set({
        messages: next,
        total:
          prevTotal === null || next.length === prev.length
            ? prevTotal
            : prevTotal - 1,
      });
      try {
        await removeMyPhotographerMessage(id);
      } catch (err) {
        console.error("[me/messages] remove failed", err);
        set({ messages: prev, total: prevTotal });
      }
    },
    applyPush: (msg) =>
      set((s) => {
        // Dedupe by id — pushes can arrive concurrently with the optimistic
        // path or be re-fanned across sessions. The newer payload wins on
        // collision (BE-side mutations like markRead also flow through the
        // REST path so push always carries server truth at emit time).
        const existingIdx = s.messages.findIndex((m) => m.id === msg.id);
        if (existingIdx !== -1) {
          const arr = s.messages.slice();
          arr[existingIdx] = msg;
          return { messages: arr };
        }
        // A genuinely new row also grows the server-side total.
        return {
          messages: [msg, ...s.messages],
          total: s.total === null ? null : s.total + 1,
        };
      }),
    setWsConnected: (connected) => set({ wsConnected: connected }),
    reset: () =>
      set({
        messages: [],
        total: null,
        limit: MESSAGES_PAGE,
        loading: false,
        error: null,
        fetchedAt: 0,
        inFlight: null,
        wsConnected: false,
      }),
  }),
);

export interface UseMyPhotographerMessagesResult {
  messages: PhotographerMessage[];
  total: number | null;
  hasMore: boolean;
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  loadMore: () => Promise<void>;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
  remove: (id: string) => Promise<void>;
}

// Subscriber hook. Pass `enabled=false` to skip the fetch — useful when
// the caller is signed out / non-photographer and the BE would 403 anyway.
export function useMyPhotographerMessages(
  enabled: boolean = true,
): UseMyPhotographerMessagesResult {
  const messages = useMyPhotographerMessagesStore((s) => s.messages);
  const total = useMyPhotographerMessagesStore((s) => s.total);
  const limit = useMyPhotographerMessagesStore((s) => s.limit);
  const loading = useMyPhotographerMessagesStore((s) => s.loading);
  const error = useMyPhotographerMessagesStore((s) => s.error);
  const fetchedAt = useMyPhotographerMessagesStore((s) => s.fetchedAt);
  const wsConnected = useMyPhotographerMessagesStore((s) => s.wsConnected);
  const refetch = useMyPhotographerMessagesStore((s) => s.refetch);
  const loadMore = useMyPhotographerMessagesStore((s) => s.loadMore);
  const markRead = useMyPhotographerMessagesStore((s) => s.markRead);
  const markAllRead = useMyPhotographerMessagesStore((s) => s.markAllRead);
  const remove = useMyPhotographerMessagesStore((s) => s.remove);

  const hasMore =
    total !== null && messages.length < total && limit < MESSAGES_MAX;

  useEffect(() => {
    if (!enabled) return;
    const staleMs = wsConnected ? STALE_HEALTHY_MS : STALE_DEGRADED_MS;
    if (fetchedAt === 0 || Date.now() - fetchedAt > staleMs) {
      void refetch();
    }
  }, [enabled, fetchedAt, wsConnected, refetch]);

  // Same focus/visibilitychange refetch the admin queue uses — covers the
  // cross-session approve loop (admin acts in another browser → this tab
  // refocuses → the new message appears without a manual refresh).
  useEffect(() => {
    if (!enabled) return;
    function onVisible() {
      if (!document.hidden) void refetch();
    }
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [enabled, refetch]);

  return {
    messages,
    total,
    hasMore,
    loading,
    error,
    refetch,
    loadMore,
    markRead,
    markAllRead,
    remove,
  };
}
