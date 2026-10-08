"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminEvents, createAdminEvent, updateAdminEvent, deleteAdminEvent,
  approveAdminEvent, rejectAdminEvent,
  type AdminEventListArgs, type AdminEventRow,
  type CreateAdminEventArgs, type UpdateAdminEventPatch,
} from "@/lib/api-admin";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { useAdminMutation } from "./use-admin-mutation";
import { claimAdminIntent, completeAdminIntent, adminIntentRecoveryNotice } from "@/lib/admin-action-intents";
import { captureAuthSession } from "@/lib/auth-session";

export const EMPTY_ADMIN_EVENTS: AdminEventRow[] = [];
export const adminEventKeys = {
  all: ["admin", "events"] as const,
  owner: (userId: string | null) => ["admin", "events", userId] as const,
  list: (userId: string | null, args: AdminEventListArgs = {}) =>
    [...adminEventKeys.owner(userId), args] as const,
};

export function useAdminEvents(args: AdminEventListArgs = {}) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminEventKeys.list(user?.id ?? null, args),
    queryFn: () => fetchAdminEvents(args),
    enabled: user?.role === "ADMIN",
    staleTime: 60_000,
  });
}

export function useAdminEventActions() {
  const qc = useQueryClient();
  const mutation = useAdminMutation();
    // Preserve rejected promises for the shared delete confirmation's inline
    // backend error. False means a stale/unauthorized session; never show success.
    const settle = async (
      label: string,
      operation: () => Promise<AdminEventRow | { removed: boolean }>,
      deletedId?: string,
    ): Promise<boolean> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || session.user?.role !== "ADMIN") return false;
      const userId = session.user.id;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated && current.user?.role === "ADMIN" &&
          current.user.id === userId && current.sessionVersion === session.sessionVersion;
      };
      const affectedKeys = [
        adminEventKeys.owner(userId), ["public", "events"], ["events"],
        ["photographer", "events"], ["admin", "kpis"], ["admin", "sales"],
      ];
      let result: AdminEventRow | { removed: boolean };
      try {
        result = await operation();
      } catch (error) {
        if (!isCurrent()) return false;
        useToastStore.getState().showToast({
          kind: "error",
          message: error instanceof ApiError ? (error.errors[0]?.message ?? error.message)
            : error instanceof Error ? error.message : `Couldn't ${label} event. Try again.`,
        });
        // A conflicting review means another admin may have decided already.
        if (error instanceof ApiError && error.status === 409) {
          void Promise.allSettled([
            qc.invalidateQueries({ queryKey: adminEventKeys.owner(userId) }),
            qc.invalidateQueries({ queryKey: ["admin", "kpis"] }),
          ]);
          if (!isCurrent()) return false;
        }
        throw error;
      }
      if (!isCurrent()) return false;
      await Promise.all(affectedKeys.map((queryKey) => qc.cancelQueries({ queryKey })));
      if (!isCurrent()) return false;
      for (const [key] of qc.getQueriesData<AdminEventRow[]>({ queryKey: adminEventKeys.owner(userId) })) {
        const args = key.at(-1) as AdminEventListArgs;
        qc.setQueryData<AdminEventRow[]>(key, (rows) => {
          if (!rows) return rows;
          // removed:false is the backend's idempotent already-deleted result.
          if (deletedId) return rows.filter((row) => row.id !== deletedId);
          if (!("id" in result)) return rows;
          const event = result;
          const matches = (!args.state || args.state === event.state) &&
            (!args.review || event.reviewStatus === "pending" || event.reviewStatus === "change_pending");
          const existing = rows.some((row) => row.id === event.id);
          if (!existing && label === "create" && matches && !args.offset) {
            const next = [event, ...rows];
            return args.limit ? next.slice(0, args.limit) : next;
          }
          return rows.flatMap((row) => row.id !== event.id ? [row] : matches ? [event] : []);
        });
      }
      // Public visibility/filter membership comes from its own endpoint. Never
      // insert private drafts or unlisted admin records into the public cache.
      void Promise.allSettled(affectedKeys.map((queryKey) => qc.invalidateQueries({ queryKey })));
      return isCurrent();
    };
    const keyed = async (operation: "event.create" | "event.update", target: string, draft: CreateAdminEventArgs | UpdateAdminEventPatch, send: (key: string) => Promise<AdminEventRow>) => {
      const current = captureAuthSession();
      const intent = await claimAdminIntent(useAuthStore.getState().user!.id, operation, target, draft);
      if (!current()) throw new Error("Session changed.");
      try {
        const row = await send(intent.key);
        if (current()) completeAdminIntent(intent);
        return row;
      } catch (error) {
        if (!current()) throw error;
        // These cover checks run before receipt claim or any storage write.
        if (error instanceof ApiError && error.errors.every((entry) => entry.field === "cover" &&
          ["VALIDATION_ERROR", "UNSUPPORTED_MEDIA_TYPE", "PAYLOAD_TOO_LARGE"].includes(entry.code))) {
          completeAdminIntent(intent);
          throw error;
        }
        const message = `${error instanceof Error ? error.message : "Request failed."} Retry the unchanged draft to check the same request.${!intent.reloadSafe ? " Recovery after reload is unavailable; keep this page open." : ""}`;
        if (error instanceof ApiError) throw new ApiError(error.errors.map((entry) => ({ ...entry, message })), error.status);
        throw new Error(message);
      }
    };
    const run = async (id: string, action: string, label: string, operation: () => Promise<AdminEventRow | { removed: boolean }>, deletedId?: string) => {
      const result = await mutation.run({ targets: [{ kind: "event", id }], action, execute: () => settle(label, operation, deletedId) });
      return result.kind === "completed" && result.value;
    };
    return {
      recoveryNotice: (id: string) => adminIntentRecoveryNotice(useAuthStore.getState().user?.id ?? "", id === "create" ? "event.create" : "event.update", id),
      isPending: (id: string) => mutation.isPending([{ kind: "event", id }]),
      pendingAction: (id: string) => mutation.pendingAction([{ kind: "event", id }]),
      createEvent: (args: CreateAdminEventArgs) => run("create", "Creating…", "create", () => keyed("event.create", "create", args, (key) => createAdminEvent(args, key))),
      editEvent: (id: string, patch: UpdateAdminEventPatch) => run(id, "Saving…", "update", () => keyed("event.update", id, patch, (key) => updateAdminEvent(id, patch, key))),
      deleteEvent: (id: string) => run(id, "Deleting…", "delete", () => deleteAdminEvent(id), id),
      approve: (id: string, revision?: number) => run(id, "Approving…", "approve", () => approveAdminEvent(id, revision)),
      reject: (id: string, reason: string, revision?: number) => run(id, "Sending back…", "reject", () => rejectAdminEvent(id, reason, revision)),
    };
}
