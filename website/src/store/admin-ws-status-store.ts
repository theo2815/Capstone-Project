"use client";

import { create } from "zustand";

// Single source of truth for the admin WebSocket's connection state.
//
// `useAdminNotificationsWs` writes here on ws.onopen ("healthy") and on
// ws.onclose + teardown ("degraded"). `useAdminQueueRealtime` reads here to
// pick its polling cadence — 5 min when WS is healthy (safety heartbeat),
// 30 s when degraded (actively compensating for the missing push channel).
//
// Initial state is "degraded" so the very first paint runs at the
// aggressive interval until the WS confirms it's connected — worst case is
// one extra refetch in the first 30 s of an admin session, which is cheaper
// than the alternative (assuming the WS is up when it actually isn't).
//
// Closes A-2.

export type AdminWsStatus = "healthy" | "degraded";

interface AdminWsStatusState {
  status: AdminWsStatus;
  setStatus: (status: AdminWsStatus) => void;
}

export const useAdminWsStatusStore = create<AdminWsStatusState>((set) => ({
  status: "degraded",
  setStatus: (status) => set({ status }),
}));
