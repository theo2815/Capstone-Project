import { create } from "zustand";
import { persist } from "zustand/middleware";
import { ADMIN_FLAGS, type Flag } from "@/lib/admin-flags";
import {
  hideAdminFlag,
  dismissAdminFlag,
  escalateAdminFlag,
  resolveAdminFlag,
} from "@/lib/api-admin";

// Store of admin actions on the moderation flag queue.
// Applies optimistic local overrides while firing real backend API calls in the background.
//
// Persisted to localStorage so flag actions survive a page refresh.
// Cleared on logout by `resetUserScopedStores()` (lib/auth-reset.ts) so cross-admin sessions don't leak decisions.

function fireBackendFlagAction(label: string, p: Promise<unknown>): void {
  void p.catch((err) => {
    console.error(`[admin/flags] ${label} backend call failed`, err);
  });
}

export type FlagDecision = "hidden" | "dismissed" | "escalated" | "resolved";

export interface FlagLogEntry {
  flagId: string;
  decision: FlagDecision;
  reason: string | null;
  decidedAt: string;
}

interface AdminFlagStoreState {
  overrides: Record<string, Partial<Flag>>;
  log: FlagLogEntry[];
  hide: (flagId: string, reason: string | null) => void;
  dismiss: (flagId: string, reason?: string | null) => void;
  escalate: (flagId: string, reason: string | null) => void;
  resolve: (flagId: string, reason: string | null) => void;
  clear: () => void;
}

function appendLog(prev: FlagLogEntry[], entry: FlagLogEntry): FlagLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

export const useAdminFlagStore = create<AdminFlagStoreState>()(
  persist(
    (set) => ({
      overrides: {},
      log: [],
      hide: (flagId, reason) => {
        set((s) => ({
          overrides: {
            ...s.overrides,
            [flagId]: {
              ...s.overrides[flagId],
              status: "hidden",
              reviewedAt: new Date().toISOString(),
              reviewedBy: "admin",
              reviewerNote: reason,
            },
          },
          log: appendLog(s.log, {
            flagId,
            decision: "hidden",
            reason,
            decidedAt: new Date().toISOString(),
          }),
        }));
        fireBackendFlagAction("hide", hideAdminFlag(flagId, reason));
      },
      dismiss: (flagId, reason = null) => {
        set((s) => ({
          overrides: {
            ...s.overrides,
            [flagId]: {
              ...s.overrides[flagId],
              status: "dismissed",
              reviewedAt: new Date().toISOString(),
              reviewedBy: "admin",
              reviewerNote: reason,
            },
          },
          log: appendLog(s.log, {
            flagId,
            decision: "dismissed",
            reason,
            decidedAt: new Date().toISOString(),
          }),
        }));
        fireBackendFlagAction("dismiss", dismissAdminFlag(flagId, reason));
      },
      escalate: (flagId, reason) => {
        set((s) => ({
          overrides: {
            ...s.overrides,
            [flagId]: {
              ...s.overrides[flagId],
              status: "escalated",
              reviewedAt: new Date().toISOString(),
              reviewedBy: "admin",
              reviewerNote: reason,
            },
          },
          log: appendLog(s.log, {
            flagId,
            decision: "escalated",
            reason,
            decidedAt: new Date().toISOString(),
          }),
        }));
        fireBackendFlagAction("escalate", escalateAdminFlag(flagId, reason));
      },
      resolve: (flagId, reason) => {
        set((s) => ({
          overrides: {
            ...s.overrides,
            [flagId]: {
              ...s.overrides[flagId],
              status: "resolved",
              reviewedAt: new Date().toISOString(),
              reviewedBy: "admin",
              reviewerNote: reason,
            },
          },
          log: appendLog(s.log, {
            flagId,
            decision: "resolved",
            reason,
            decidedAt: new Date().toISOString(),
          }),
        }));
        fireBackendFlagAction("resolve", resolveAdminFlag(flagId, reason));
      },
      clear: () => set({ overrides: {}, log: [] }),
    }),
    {
      name: "quickpitik-admin-flag",
      partialize: (s) => ({ overrides: s.overrides, log: s.log }),
    },
  ),
);

export function mergeFlag(
  base: Flag,
  patch: Partial<Flag> | undefined,
): Flag {
  return patch ? { ...base, ...patch } : base;
}

export function getEffectiveFlags(
  overrides: Record<string, Partial<Flag>> = {},
): Flag[] {
  return ADMIN_FLAGS.map((f) => mergeFlag(f, overrides[f.id]));
}

/**
 * Merges live server data with local overrides for instant feedback after admin actions.
 */
export function mergeFlagsWithOverrides(
  serverData: ReadonlyArray<Flag>,
  overrides: Record<string, Partial<Flag>> = {},
): Flag[] {
  return serverData.map((f) => mergeFlag(f, overrides[f.id]));
}
