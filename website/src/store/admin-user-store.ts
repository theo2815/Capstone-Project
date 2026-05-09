import { create } from "zustand";
import {
  ADMIN_USER_SEED,
  type AdminUserRow,
} from "@/lib/admin-user-registry";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  approveUser as apiApproveUser,
  rejectUser as apiRejectUser,
  resetUserVerification as apiResetUserVerification,
  suspendUser as apiSuspendUser,
  unsuspendUser as apiUnsuspendUser,
  forceEditUser as apiForceEditUser,
} from "@/lib/api-admin";

// Mock-mode store of admin actions on the user directory. Phase G wiring
// fires the matching `api-admin` call in the background after the local
// override is applied — Q-A2 RESOLVED 2026-05-09 means the backend writes
// the cross-domain side-effect (photographer-message insert) in the same
// TX, so the FE only needs to fire the action and let the photographer's
// next inbox poll surface the new row. Local override gives the admin UI
// instant feedback; backend errors log to console for now (rollback is a
// follow-up if errors surface in real usage).
//
// CRITICAL DEMO WIRING — `approve(userId)`, `reject(userId, reason)`, and
// `resetVerification(userId)` ALSO flip
// `usePhotographerSettingsStore.verificationStatus` when the target user is
// the currently-logged-in photographer. That's what makes the admin →
// photographer dashboard banner loop close end-to-end without server roundtrip.

function fireBackendUserAction(label: string, p: Promise<unknown>): void {
  if (!BACKEND_LIVE) return;
  void p.catch((err) => {
    console.error(`[admin/users] ${label} backend call failed`, err);
  });
}

export type DecisionType =
  | "approved"
  | "rejected"
  | "suspended"
  | "unsuspended"
  | "force-edited"
  | "reset";

export interface DecisionLogEntry {
  userId: string;
  decision: DecisionType;
  reason: string | null;
  decidedAt: string;
  /** Free-form payload — currently used by force-edit to record `field`,
   * `from`, `to` so the activity timeline can show "brand: Old → New". */
  meta?: { field?: string; from?: string; to?: string };
}

export type ForceEditPatch = Partial<
  Pick<AdminUserRow, "handle" | "brandName">
>;

interface AdminUserStoreState {
  overrides: Record<string, Partial<AdminUserRow>>;
  log: DecisionLogEntry[];
  approve: (userId: string) => void;
  reject: (userId: string, reason: string) => void;
  suspend: (userId: string, reason: string) => void;
  unsuspend: (userId: string) => void;
  forceEdit: (userId: string, patch: ForceEditPatch) => void;
  resetVerification: (userId: string) => void;
  clear: () => void;
  // Selectors
  getRow: (userId: string) => AdminUserRow | undefined;
  getEffectiveSeed: () => ReadonlyArray<AdminUserRow>;
  getPendingPhotographers: () => AdminUserRow[];
  getIncompletePhotographers: () => AdminUserRow[];
}

function mergeRow(
  base: AdminUserRow,
  patch: Partial<AdminUserRow> | undefined,
): AdminUserRow {
  return patch ? { ...base, ...patch } : base;
}

function maybeFlipPhotographerOwnStatus(
  userId: string,
  next: "approved" | "incomplete",
) {
  const sessionUser = useAuthStore.getState().user;
  if (!sessionUser || sessionUser.id !== userId) return;
  // The logged-in photographer was approved/rejected/reset — flip their own
  // settings store so the dashboard banner reflects it without a refresh.
  usePhotographerSettingsStore.getState().setVerificationStatus(next);
}

function appendLog(
  prev: DecisionLogEntry[],
  entry: DecisionLogEntry,
): DecisionLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

export const useAdminUserStore = create<AdminUserStoreState>((set, get) => ({
  overrides: {},
  log: [],
  approve: (userId) => {
    set((s) => ({
      overrides: {
        ...s.overrides,
        [userId]: {
          ...s.overrides[userId],
          verificationStatus: "approved",
        },
      },
      log: appendLog(s.log, {
        userId,
        decision: "approved",
        reason: null,
        decidedAt: new Date().toISOString(),
      }),
    }));
    maybeFlipPhotographerOwnStatus(userId, "approved");
    fireBackendUserAction("approve", apiApproveUser(userId));
  },
  reject: (userId, reason) => {
    set((s) => ({
      overrides: {
        ...s.overrides,
        [userId]: {
          ...s.overrides[userId],
          verificationStatus: "incomplete",
        },
      },
      log: appendLog(s.log, {
        userId,
        decision: "rejected",
        reason,
        decidedAt: new Date().toISOString(),
      }),
    }));
    maybeFlipPhotographerOwnStatus(userId, "incomplete");
    fireBackendUserAction("reject", apiRejectUser(userId, reason));
  },
  suspend: (userId, reason) => {
    set((s) => ({
      overrides: {
        ...s.overrides,
        [userId]: {
          ...s.overrides[userId],
          suspendedAt: new Date().toISOString(),
          suspensionReason: reason,
        },
      },
      log: appendLog(s.log, {
        userId,
        decision: "suspended",
        reason,
        decidedAt: new Date().toISOString(),
      }),
    }));
    fireBackendUserAction("suspend", apiSuspendUser(userId, reason));
  },
  unsuspend: (userId) => {
    set((s) => ({
      overrides: {
        ...s.overrides,
        [userId]: {
          ...s.overrides[userId],
          suspendedAt: null,
          suspensionReason: null,
        },
      },
      log: appendLog(s.log, {
        userId,
        decision: "unsuspended",
        reason: null,
        decidedAt: new Date().toISOString(),
      }),
    }));
    fireBackendUserAction("unsuspend", apiUnsuspendUser(userId));
  },
  forceEdit: (userId, patch) => {
    const current = get().getRow(userId);
    if (!current) return;
    // Build one log entry per changed field so the activity timeline reads
    // each change discretely. Skip no-op changes.
    const changes: DecisionLogEntry[] = [];
    const now = new Date().toISOString();
    if (patch.handle !== undefined && patch.handle !== current.handle) {
      changes.push({
        userId,
        decision: "force-edited",
        reason: null,
        decidedAt: now,
        meta: {
          field: "handle",
          from: current.handle ?? "—",
          to: patch.handle ?? "—",
        },
      });
    }
    if (
      patch.brandName !== undefined &&
      patch.brandName !== current.brandName
    ) {
      changes.push({
        userId,
        decision: "force-edited",
        reason: null,
        decidedAt: now,
        meta: {
          field: "brandName",
          from: current.brandName ?? "—",
          to: patch.brandName ?? "—",
        },
      });
    }
    if (changes.length === 0) return;
    set((s) => {
      let log = s.log;
      for (const change of changes) {
        log = appendLog(log, change);
      }
      return {
        overrides: {
          ...s.overrides,
          [userId]: {
            ...s.overrides[userId],
            ...patch,
          },
        },
        log,
      };
    });
    fireBackendUserAction("forceEdit", apiForceEditUser(userId, patch));
  },
  resetVerification: (userId) => {
    set((s) => ({
      overrides: {
        ...s.overrides,
        [userId]: {
          ...s.overrides[userId],
          verificationStatus: "incomplete",
        },
      },
      log: appendLog(s.log, {
        userId,
        decision: "reset",
        reason: null,
        decidedAt: new Date().toISOString(),
      }),
    }));
    maybeFlipPhotographerOwnStatus(userId, "incomplete");
    fireBackendUserAction("resetVerification", apiResetUserVerification(userId));
  },
  clear: () => set({ overrides: {}, log: [] }),
  getRow: (userId) => {
    const base = ADMIN_USER_SEED.find((u) => u.userId === userId);
    if (!base) return undefined;
    return mergeRow(base, get().overrides[userId]);
  },
  getEffectiveSeed: () => {
    const { overrides } = get();
    if (Object.keys(overrides).length === 0) return ADMIN_USER_SEED;
    return ADMIN_USER_SEED.map((row) => mergeRow(row, overrides[row.userId]));
  },
  getPendingPhotographers: () =>
    get()
      .getEffectiveSeed()
      .filter(
        (u) => u.role === "PHOTOGRAPHER" && u.verificationStatus === "pending",
      ),
  getIncompletePhotographers: () =>
    get()
      .getEffectiveSeed()
      .filter(
        (u) =>
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "incomplete",
      ),
}));
