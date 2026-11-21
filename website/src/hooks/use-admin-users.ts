"use client";

import { useQuery, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import type { PaginatedResponse } from "@/types/api";
import {
  fetchAdminUsers, fetchAdminUsersPage, fetchAdminUserDetail, fetchAdminPhotographerSettings, fetchAdminUserRecord,
  approveUser, rejectUser, suspendUser, unsuspendUser, resetUserVerification,
  sendAdminMessage, fetchAdminCommand, type SendAdminMessageInput,
  type AdminUserDetail, type AdminUserStatusFilter,
} from "@/lib/api-admin";
import { useInfiniteList } from "./use-infinite-list";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { useAdminMutation } from "./use-admin-mutation";
import { claimAdminIntent, completeAdminIntent, adminIntentRecoveryNotice } from "@/lib/admin-action-intents";
import { captureAuthSession } from "@/lib/auth-session";

export const EMPTY_ADMIN_USERS: AdminUserRow[] = [];
export const adminUserKeys = {
  all: ["admin", "users"] as const,
  owner: (owner: string | null) => ["admin", "users", owner] as const,
  list: (owner: string | null) => [...adminUserKeys.owner(owner), "list"] as const,
  detail: (owner: string | null, id: string | null) => [...adminUserKeys.owner(owner), "detail", id] as const,
  settings: (owner: string | null, id: string | null) => [...adminUserKeys.owner(owner), "settings", id] as const,
  record: (owner: string | null, id: string | null) => [...adminUserKeys.owner(owner), "record", id] as const,
};

export function useAdminUsers() {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminUserKeys.list(user?.id ?? null),
    queryFn: ({ signal }) => fetchAdminUsers({}, signal),
    enabled: user?.role === "ADMIN",
    staleTime: 30_000,
  });
}

// Under the owner prefix so every user mutation's refresh reaches it; nothing patches that prefix as an array.
export function useAdminPhotographersPage(status: AdminUserStatusFilter, q: string) {
  const user = useAuthStore((s) => s.user);
  return useInfiniteList<AdminUserRow & { id: string }>({
    queryKey: [...adminUserKeys.owner(user?.id ?? null), "photographers", status, q],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchAdminUsersPage({ role: "PHOTOGRAPHER", status, q: q || undefined, offset, limit }, signal);
      return { ...page, items: page.items.map((row) => ({ ...row, id: row.userId })) };
    },
    limit: 50,
    enabled: user?.role === "ADMIN",
    staleTime: 30_000,
  });
}

// AX-41: same owner prefix as the photographers list, so suspend and unsuspend refresh it.
export function useAdminRunnersPage(q: string) {
  const user = useAuthStore((s) => s.user);
  return useInfiniteList<AdminUserRow & { id: string }>({
    queryKey: [...adminUserKeys.owner(user?.id ?? null), "runners", q],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchAdminUsersPage({ role: "RUNNER", q: q || undefined, offset, limit }, signal);
      return { ...page, items: page.items.map((row) => ({ ...row, id: row.userId })) };
    },
    limit: 50,
    enabled: user?.role === "ADMIN",
    staleTime: 30_000,
  });
}

export function useAdminUserDetail(id: string | null) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminUserKeys.detail(user?.id ?? null, id),
    queryFn: async () => {
      try { return await fetchAdminUserDetail(id!); }
      catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    enabled: user?.role === "ADMIN" && !!id,
    staleTime: 30_000,
  });
}

export function useAdminPhotographerSettings(id: string | null) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminUserKeys.settings(user?.id ?? null, id),
    queryFn: () => fetchAdminPhotographerSettings(id!),
    enabled: user?.role === "ADMIN" && !!id,
    staleTime: 60_000,
  });
}

// AX-27: under the owner prefix, so every user mutation's refresh reaches it; null when an older backend has no route.
export function useAdminUserRecord(id: string | null) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: adminUserKeys.record(user?.id ?? null, id),
    queryFn: async ({ signal }) => {
      try { return await fetchAdminUserRecord(id!, signal); }
      catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
    enabled: user?.role === "ADMIN" && !!id,
    staleTime: 60_000,
  });
}

/** Null until the list has loaded, so a header never shows a loading or failed 0. */
export function usePendingVerificationsCount(): number | null {
  const rows = useAdminUsers().data;
  if (!rows) return null;
  return rows.filter((row) => row.role === "PHOTOGRAPHER" &&
    row.verificationStatus === "pending" && row.suspendedAt === null).length;
}

export function useAdminUserActions() {
  const qc = useQueryClient();
  const mutation = useAdminMutation();
  const refresh = (owner: string) => {
    void Promise.allSettled([adminUserKeys.owner(owner), ["admin", "kpis"]].map((queryKey) => qc.invalidateQueries({ queryKey })));
  };
    const settle = async (id: string, label: string, operation: () => Promise<AdminUserRow>, batch = false): Promise<boolean> => {
      const session = useAuthStore.getState();
      if (!session.isAuthenticated || session.user?.role !== "ADMIN") return false;
      const owner = session.user.id;
      const isCurrent = () => {
        const current = useAuthStore.getState();
        return current.isAuthenticated && current.user?.role === "ADMIN" &&
          current.user.id === owner && current.sessionVersion === session.sessionVersion;
      };
      if (label === "suspend" && owner === id) {
        useToastStore.getState().showToast({ kind: "error", message: "You can't suspend your own admin account." });
        return false;
      }
      let row: AdminUserRow;
      try { row = await operation(); }
      catch (error) {
        if (!isCurrent()) return false;
        useToastStore.getState().showToast({ kind: "error", message:
          error instanceof ApiError ? (error.errors[0]?.field === "expectedRevision" ? "This submission changed. Close this form and review the current submission before deciding again." : error.errors[0]?.message ?? error.message)
            : `Couldn't ${label} — please try again.`,
        });
        if (error instanceof ApiError && error.status === 409) {
          if (!batch) refresh(owner);
        }
        return false;
      }
      if (!isCurrent()) return false;
      const keys = [adminUserKeys.owner(owner), ["admin", "kpis"]];
      await Promise.all(keys.map((queryKey) => qc.cancelQueries({ queryKey })));
      if (!isCurrent()) return false;
      qc.setQueryData<AdminUserRow[]>(adminUserKeys.list(owner), (rows) =>
        rows?.map((existing) => existing.userId === id ? row : existing));
      // AX-65: the paged lists too, so a row's Suspend turns to Unsuspend before the refetch lands.
      // AX-69: a photographer tab the row no longer matches drops it; the refetch adds it to its new tab.
      const status = row.suspendedAt ? "suspended" : row.verificationStatus;
      for (const [key, data] of qc.getQueriesData<InfiniteData<PaginatedResponse<AdminUserRow & { id: string }>>>(
        { queryKey: adminUserKeys.owner(owner), predicate: (query) => query.queryKey[3] === "photographers" || query.queryKey[3] === "runners" })) {
        if (!data?.pages.some((page) => page.items.some((existing) => existing.userId === id))) continue;
        const leaves = key[3] === "photographers" && key[4] !== status;
        qc.setQueryData(key, { ...data, pages: data.pages.map((page) => leaves
          ? { ...page, total: page.total - 1, items: page.items.filter((existing) => existing.userId !== id) }
          : { ...page, items: page.items.map((existing) => existing.userId === id ? { ...row, id } : existing) }) });
      }
      qc.setQueryData<AdminUserDetail | null>(adminUserKeys.detail(owner, id), (detail) =>
        detail ? { ...row, decisionLog: detail.decisionLog,
          // A list row never carries the withdrawal (RV-L7); only an explicit null from the server clears it.
          relevanceConsentWithdrawnAt: row.relevanceConsentWithdrawnAt === undefined ? detail.relevanceConsentWithdrawnAt : row.relevanceConsentWithdrawnAt } : detail);
      // The mutation returns a row, not history or settings. Refetch those
      // resources; never manufacture a decision or timestamp in the browser.
      if (!batch) refresh(owner);
      return isCurrent();
    };
    const run = async (id: string, action: string, operation: () => Promise<AdminUserRow>, verb = action) => {
      const result = await mutation.run({ targets: [{ kind: "user", id }], action, execute: () => settle(id, verb, operation) });
      return result.kind === "completed" && result.value;
    };
    return {
      messageRecoveryNotice: (id: string) => adminIntentRecoveryNotice(useAuthStore.getState().user?.id ?? "", "user.message", id),
      message: async (id: string, input: SendAdminMessageInput) => {
        const result = await mutation.run({ targets: [{ kind: "user", id }], action: "Sending…", execute: async () => {
          const current = captureAuthSession();
          const owner = useAuthStore.getState().user!.id;
          let intent: Awaited<ReturnType<typeof claimAdminIntent>> | undefined;
          try {
            intent = await claimAdminIntent(owner, "user.message", id, input, async (key) => {
              const command = await fetchAdminCommand(key);
              if (!command) return "missing";
              if (command.state !== "SUCCEEDED" || !current()) return "open";
              useToastStore.getState().showToast({ kind: "success", duration: 10_000, message: "Your earlier message to this user was delivered. It is in their inbox." });
              return "succeeded";
            });
            if (!current()) return false;
            await sendAdminMessage(id, input, intent.key);
            if (!current()) return false;
            completeAdminIntent(intent); refresh(owner); return true;
          } catch (error) {
            if (!current()) return false;
            // A first attempt the server refused outright wrote nothing: the receipt and the message commit together.
            if (intent && !intent.reused && error instanceof ApiError && [400, 403, 404, 422].includes(error.status ?? 0)) {
              completeAdminIntent(intent);
              useToastStore.getState().showToast({ kind: "error", message: error.message });
              return false;
            }
            useToastStore.getState().showToast({ kind: "error", message: `${error instanceof Error ? error.message : "Couldn't send message."} Retry the unchanged draft to check the same request.${intent?.reloadSafe === false ? " Recovery after reload is unavailable; keep this page open." : ""}` });
            return false;
          }
        } });
        return result.kind === "completed" && result.value;
      },
      isPending: (ids: string[]) => mutation.isPending(ids.map((id) => ({ kind: "user", id }))),
      pendingAction: (id: string) => mutation.pendingAction([{ kind: "user", id }]),
      reject: (id: string, reason: string, revision?: number) => run(id, "Sending back…", () => rejectUser(id, reason, revision), "reject"),
      // Only for a profile held after a reset or reject; the server answers 409 otherwise.
      approveHeld: (id: string, revision?: number) => run(id, "Approving…", () => approveUser(id, revision), "approve"),
      suspend: (id: string, reason: string) => run(id, "suspend", () => suspendUser(id, reason)),
      unsuspend: (id: string) => run(id, "Unsuspending…", () => unsuspendUser(id), "unsuspend"),
      resetVerification: (id: string, reason: string, revision?: number) => run(id, "Resetting verification…", () => resetUserVerification(id, reason, revision), "reset verification"),
      // Bulk send back only: profiles approve themselves once identity passes (Q7).
      verifyBatch: async (rows: { userId: string; verificationRevision: number }[], reason: string): Promise<boolean[] | null> => {
        const result = await mutation.run({ targets: rows.map((row) => ({ kind: "user", id: row.userId })), action: "Sending back…", execute: async () => {
          const session = useAuthStore.getState();
          const outcomes = await Promise.all(rows.map((row) => settle(row.userId, "reject",
            () => rejectUser(row.userId, reason, row.verificationRevision), true)));
          if (useAuthStore.getState().sessionVersion === session.sessionVersion) refresh(session.user!.id);
          return outcomes;
        } });
        return result.kind === "completed" ? result.value : null;
      },
    };
}
