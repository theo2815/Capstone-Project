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

interface MessagesState {
  messages: RunnerMessage[];
  loading: boolean;
  error: string | null;
  fetchedAt: number;
  inFlight: Promise<void> | null;
  wsConnected: boolean;
  refetch: () => Promise<void>;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
  remove: (id: string) => Promise<void>;
  applyPush: (msg: RunnerMessage) => void;
  setWsConnected: (connected: boolean) => void;
  reset: () => void;
}

export const useMyRunnerMessagesStore = create<MessagesState>((set, get) => ({
  messages: [],
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
        const messages = await fetchMyRunnerMessages();
        set({
          messages,
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
    set({ messages: prev.filter((m) => m.id !== id) });
    try {
      await removeMyRunnerMessage(id);
    } catch (err) {
      console.error("[me/runner/messages] remove failed", err);
      set({ messages: prev });
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
      return { messages: [msg, ...s.messages] };
    }),
  setWsConnected: (connected) => set({ wsConnected: connected }),
  reset: () =>
    set({
      messages: [],
      loading: false,
      error: null,
      fetchedAt: 0,
      inFlight: null,
      wsConnected: false,
    }),
}));

export interface UseMyRunnerMessagesResult {
  messages: RunnerMessage[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
  markRead: (id: string) => Promise<void>;
  markAllRead: () => Promise<void>;
  remove: (id: string) => Promise<void>;
}

export function useMyRunnerMessages(
  enabled: boolean = true,
): UseMyRunnerMessagesResult {
  const messages = useMyRunnerMessagesStore((s) => s.messages);
  const loading = useMyRunnerMessagesStore((s) => s.loading);
  const error = useMyRunnerMessagesStore((s) => s.error);
  const fetchedAt = useMyRunnerMessagesStore((s) => s.fetchedAt);
  const wsConnected = useMyRunnerMessagesStore((s) => s.wsConnected);
  const refetch = useMyRunnerMessagesStore((s) => s.refetch);
  const markRead = useMyRunnerMessagesStore((s) => s.markRead);
  const markAllRead = useMyRunnerMessagesStore((s) => s.markAllRead);
  const remove = useMyRunnerMessagesStore((s) => s.remove);

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

  return { messages, loading, error, refetch, markRead, markAllRead, remove };
}
