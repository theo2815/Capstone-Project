import { create } from "zustand";
import { ADMIN_FLAGS, type Flag } from "@/lib/admin-flags";

// Mock-only store of admin actions on the moderation flag queue. NOT
// persisted — real backend ships in Phase F. Same shape as
// admin-dispute-store: overrides + capped log + per-action mutators.

export type FlagDecision = "hidden" | "dismissed" | "escalated";

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
  dismiss: (flagId: string) => void;
  escalate: (flagId: string, reason: string | null) => void;
  clear: () => void;
}

function appendLog(prev: FlagLogEntry[], entry: FlagLogEntry): FlagLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

export const useAdminFlagStore = create<AdminFlagStoreState>((set) => ({
  overrides: {},
  log: [],
  hide: (flagId, reason) =>
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
    })),
  dismiss: (flagId) =>
    set((s) => ({
      overrides: {
        ...s.overrides,
        [flagId]: {
          ...s.overrides[flagId],
          status: "dismissed",
          reviewedAt: new Date().toISOString(),
          reviewedBy: "admin",
          reviewerNote: null,
        },
      },
      log: appendLog(s.log, {
        flagId,
        decision: "dismissed",
        reason: null,
        decidedAt: new Date().toISOString(),
      }),
    })),
  escalate: (flagId, reason) =>
    set((s) => ({
      overrides: {
        ...s.overrides,
        [flagId]: {
          ...s.overrides[flagId],
          status: "escalated",
        },
      },
      log: appendLog(s.log, {
        flagId,
        decision: "escalated",
        reason,
        decidedAt: new Date().toISOString(),
      }),
    })),
  clear: () => set({ overrides: {}, log: [] }),
}));

export function mergeFlag(
  base: Flag,
  patch: Partial<Flag> | undefined,
): Flag {
  return patch ? { ...base, ...patch } : base;
}

export function getEffectiveFlags(
  overrides: Record<string, Partial<Flag>>,
): Flag[] {
  return ADMIN_FLAGS.map((f) => mergeFlag(f, overrides[f.id]));
}
