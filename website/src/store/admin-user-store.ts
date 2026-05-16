import { create } from "zustand";
import { persist } from "zustand/middleware";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useToastStore } from "@/store/toast-store";
import { useAdminUsersServerStore } from "@/lib/admin-users-data";
// F-NEW-1a — every mutating action calls refetchAdminPhotographerSettings
// after upsertRow so any open drawer or detail page sees fresh BE state
// without waiting out the 60s settings-cache TTL. Skipped automatically
// when the userId isn't cached (no observer to update).
import { refetchAdminPhotographerSettings } from "@/lib/admin-photographer-settings-data";
import {
  approveUser as apiApproveUser,
  rejectUser as apiRejectUser,
  resetUserVerification as apiResetUserVerification,
  suspendUser as apiSuspendUser,
  unsuspendUser as apiUnsuspendUser,
  forceEditUser as apiForceEditUser,
} from "@/lib/api-admin";

// Phase 3 — admin actions now await the BE and write the returned row into
// the `useAdminUsersServerStore` cache. The optimistic-overrides slice is
// gone: BE is the source of truth, and the action waits for it to confirm.
//
// On failure, the photographer-settings flip (if it was the logged-in
// photographer) is rolled back to its prior status and a toast surfaces
// the error. The decision log entry is only appended on success — keeping
// the log in sync with the BE's persisted decisions.
//
// DEMO WIRING — `approve`, `reject`, `resetVerification` ALSO flip
// `usePhotographerSettingsStore.verificationStatus` when the target user
// is the logged-in photographer (single-browser flow). The cross-session
// case is Phase 4's concern (focus-refetch).

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
  /** Photographer rows that originated from a real self-submit. Kept as a
   *  transient client-side optimistic insertion so the row shows up in the
   *  admin queue immediately in the demo flow before the 30s revalidation
   *  pulls it from the BE. Filtered against server rows on read to avoid
   *  double-rendering. Phase 5 may remove once Phase 4 wires focus-refetch. */
  submissions: Record<string, AdminUserRow>;
  log: DecisionLogEntry[];
  approve: (userId: string) => Promise<void>;
  reject: (userId: string, reason: string) => Promise<void>;
  suspend: (userId: string, reason: string) => Promise<void>;
  unsuspend: (userId: string) => Promise<void>;
  forceEdit: (userId: string, patch: ForceEditPatch) => Promise<void>;
  resetVerification: (userId: string) => Promise<void>;
  upsertSubmission: (row: AdminUserRow) => void;
  revokeSubmission: (userId: string) => void;
  clear: () => void;
}

function appendLog(
  prev: DecisionLogEntry[],
  entry: DecisionLogEntry,
): DecisionLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

// Capture the photographer's prior status before flipping, so we can roll
// back if the BE call fails. Returns null when the target isn't the
// currently-logged-in photographer (no flip happened, nothing to roll back).
function flipPhotographerOwnStatus(
  userId: string,
  next: "approved" | "incomplete",
): { rollback: () => void } | null {
  const sessionUser = useAuthStore.getState().user;
  if (!sessionUser || sessionUser.id !== userId) return null;
  const settings = usePhotographerSettingsStore.getState();
  const prev = settings.verificationStatus;
  settings.setVerificationStatus(next);
  return {
    rollback: () =>
      usePhotographerSettingsStore.getState().setVerificationStatus(prev),
  };
}

function toastBackendFailure(label: string, err: unknown): void {
  console.error(`[admin/users] ${label} failed`, err);
  useToastStore.getState().showToast({
    kind: "error",
    message: `Couldn't ${label} — please try again.`,
  });
}

export const useAdminUserStore = create<AdminUserStoreState>()(
  persist(
    (set) => ({
      submissions: {},
      log: [],
      approve: async (userId) => {
        const flip = flipPhotographerOwnStatus(userId, "approved");
        try {
          const updated = await apiApproveUser(userId);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => ({
            log: appendLog(s.log, {
              userId,
              decision: "approved",
              reason: null,
              decidedAt: new Date().toISOString(),
            }),
          }));
        } catch (err) {
          flip?.rollback();
          toastBackendFailure("approve", err);
        }
      },
      reject: async (userId, reason) => {
        const flip = flipPhotographerOwnStatus(userId, "incomplete");
        try {
          const updated = await apiRejectUser(userId, reason);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => ({
            log: appendLog(s.log, {
              userId,
              decision: "rejected",
              reason,
              decidedAt: new Date().toISOString(),
            }),
          }));
        } catch (err) {
          flip?.rollback();
          toastBackendFailure("reject", err);
        }
      },
      suspend: async (userId, reason) => {
        try {
          const updated = await apiSuspendUser(userId, reason);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => ({
            log: appendLog(s.log, {
              userId,
              decision: "suspended",
              reason,
              decidedAt: new Date().toISOString(),
            }),
          }));
        } catch (err) {
          toastBackendFailure("suspend", err);
        }
      },
      unsuspend: async (userId) => {
        try {
          const updated = await apiUnsuspendUser(userId);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => ({
            log: appendLog(s.log, {
              userId,
              decision: "unsuspended",
              reason: null,
              decidedAt: new Date().toISOString(),
            }),
          }));
        } catch (err) {
          toastBackendFailure("unsuspend", err);
        }
      },
      forceEdit: async (userId, patch) => {
        // Pull current row from the server cache so the log entries can
        // show "from → to" without re-fetching.
        const current = useAdminUsersServerStore
          .getState()
          .rows.find((r) => r.userId === userId);
        if (!current) {
          toastBackendFailure(
            "edit",
            new Error(`No cached row for ${userId}`),
          );
          return;
        }
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
        try {
          const updated = await apiForceEditUser(userId, patch);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => {
            let log = s.log;
            for (const change of changes) {
              log = appendLog(log, change);
            }
            return { log };
          });
        } catch (err) {
          toastBackendFailure("edit", err);
        }
      },
      resetVerification: async (userId) => {
        const flip = flipPhotographerOwnStatus(userId, "incomplete");
        try {
          const updated = await apiResetUserVerification(userId);
          useAdminUsersServerStore.getState().upsertRow(updated);
          refetchAdminPhotographerSettings(userId);
          set((s) => ({
            log: appendLog(s.log, {
              userId,
              decision: "reset",
              reason: null,
              decidedAt: new Date().toISOString(),
            }),
          }));
        } catch (err) {
          flip?.rollback();
          toastBackendFailure("reset verification", err);
        }
      },
      upsertSubmission: (row) => {
        set((s) => ({
          submissions: { ...s.submissions, [row.userId]: row },
        }));
      },
      revokeSubmission: (userId) => {
        set((s) => {
          const submissions = { ...s.submissions };
          delete submissions[userId];
          return { submissions };
        });
      },
      clear: () => set({ submissions: {}, log: [] }),
    }),
    {
      // Persist the cross-session state only — actions are re-created on
      // every store init. The `overrides` field was dropped in Phase 3:
      // the BE is now the source of truth and writes land in
      // useAdminUsersServerStore.upsertRow instead.
      name: "quickpitik-admin-user-store",
      partialize: (state) => ({
        submissions: state.submissions,
        log: state.log,
      }),
    },
  ),
);
