"use client";

import { useEffect } from "react";
import { create } from "zustand";
import type { RunnerMessage } from "@/lib/runner-messages";
import {
  fetchMyRunnerMessages,
  markAllMyRunnerMessagesRead,
  markMyRunnerMessageRead,
  removeMyRunnerMessage,
} from "@/lib/api-me-runner-messages";

// Runner-side inbox cache + actions. Mirrors the photographer pattern
// (me-photographer-messages-data.ts) — in-flight dedup, focus refetch,
// optimistic mutations, WS-driven adaptive polling. Kept as a parallel
// store so the photographer + runner inboxes can't bleed into each other
// when a single browser is signed in across sessions.

const STALE_HEALTHY_MS = 5 * 60_000;
const STALE_DEGRADED_MS = 30_000;
// The BE inbox defaults to 100 rows and caps at 200. Load-more bumps the
// requested limit and re-fetches (the store mixes WS-pushed + optimistically
// mutated rows, so a full replace is safer than offset-appending).
const MESSAGES_PAGE = 100;
const MESSAGES_MAX = 200;

interface MessagesState {
  messages: RunnerMessage[];
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
  applyPush: (msg: RunnerMessage) => void;
  setWsConnected: (connected: boolean) => void;
  reset: () => void;
}

export const useMyRunnerMessagesStore = create<MessagesState>((set, get) => ({
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
        const { messages, total } = await fetchMyRunnerMessages(get().limit);
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
          error: err instanceof Error ? err.message : "Failed to load inbox",
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
    const prev = get().messages;
    const idx = prev.findIndex((m) => m.id === id);
    if (idx === -1) return;
    if (prev[idx].readAt !== null) return;
    const at = new Date().toISOString();
    const next = prev.slice();
    next[idx] = { ...prev[idx], readAt: at };
    set({ messages: next });
    try {
      const updated = await markMyRunnerMessageRead(id);
      set((s) => {
        const i = s.messages.findIndex((m) => m.id === id);
        if (i === -1) return s;
        const arr = s.messages.slice();
        arr[i] = updated;
        return { messages: arr };
      });
    } catch (err) {
      console.error("[me/runner/messages] markRead failed", err);
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
      await markAllMyRunnerMessagesRead();
    } catch (err) {
      console.error("[me/runner/messages] markAllRead failed", err);
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
      await removeMyRunnerMessage(id);
    } catch (err) {
      console.error("[me/runner/messages] remove failed", err);
      set({ messages: prev, total: prevTotal });
    }
  },
  applyPush: (msg) =>
    set((s) => {
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
}));

export interface UseMyRunnerMessagesResult {
  messages: RunnerMessage[];
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

export function useMyRunnerMessages(
  enabled: boolean = true,
): UseMyRunnerMessagesResult {
  const messages = useMyRunnerMessagesStore((s) => s.messages);
  const total = useMyRunnerMessagesStore((s) => s.total);
  const limit = useMyRunnerMessagesStore((s) => s.limit);
  const loading = useMyRunnerMessagesStore((s) => s.loading);
  const error = useMyRunnerMessagesStore((s) => s.error);
  const fetchedAt = useMyRunnerMessagesStore((s) => s.fetchedAt);
  const wsConnected = useMyRunnerMessagesStore((s) => s.wsConnected);
  const refetch = useMyRunnerMessagesStore((s) => s.refetch);
  const loadMore = useMyRunnerMessagesStore((s) => s.loadMore);
  const markRead = useMyRunnerMessagesStore((s) => s.markRead);
  const markAllRead = useMyRunnerMessagesStore((s) => s.markAllRead);
  const remove = useMyRunnerMessagesStore((s) => s.remove);

  const hasMore =
    total !== null && messages.length < total && limit < MESSAGES_MAX;

  useEffect(() => {
    if (!enabled) return;
    const staleMs = wsConnected ? STALE_HEALTHY_MS : STALE_DEGRADED_MS;
    if (fetchedAt === 0 || Date.now() - fetchedAt > staleMs) {
      void refetch();
    }
  }, [enabled, fetchedAt, wsConnected, refetch]);

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
