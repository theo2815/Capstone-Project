"use client";

import { useMemo } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchAdminUsers, fetchAdminUserDetail, fetchAdminPhotographerSettings,
  approveUser, rejectUser, suspendUser, unsuspendUser, resetUserVerification,
  type AdminUserDetail,
} from "@/lib/api-admin";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

export const EMPTY_ADMIN_USERS: AdminUserRow[] = [];
export const adminUserKeys = {
  all: ["admin", "users"] as const,
  owner: (owner: string | null) => ["admin", "users", owner] as const,
  list: (owner: string | null) => [...adminUserKeys.owner(owner), "list"] as const,
  detail: (owner: string | null, id: string | null) => [...adminUserKeys.owner(owner), "detail", id] as const,
  settings: (owner: string | null, id: string | null) => [...adminUserKeys.owner(owner), "settings", id] as const,
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

export function usePendingVerificationsCount() {
  const rows = useAdminUsers().data ?? EMPTY_ADMIN_USERS;
  return rows.filter((row) => row.role === "PHOTOGRAPHER" &&
    row.verificationStatus === "pending" && row.suspendedAt === null).length;
}

export function useAdminUserActions() {
  const qc = useQueryClient();
  return useMemo(() => {
    const settle = async (id: string, label: string, operation: () => Promise<AdminUserRow>): Promise<boolean> => {
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
          error instanceof ApiError ? (error.errors[0]?.message ?? error.message)
            : `Couldn't ${label} — please try again.`,
        });
        if (error instanceof ApiError && error.status === 409) {
          await qc.invalidateQueries({ queryKey: adminUserKeys.owner(owner) });
        }
        return false;
      }
      if (!isCurrent()) return false;
      const keys = [adminUserKeys.owner(owner), ["admin", "kpis"]];
      await Promise.all(keys.map((queryKey) => qc.cancelQueries({ queryKey })));
      if (!isCurrent()) return false;
      qc.setQueryData<AdminUserRow[]>(adminUserKeys.list(owner), (rows) =>
        rows?.map((existing) => existing.userId === id ? row : existing));
      qc.setQueryData<AdminUserDetail | null>(adminUserKeys.detail(owner, id), (detail) =>
        detail ? { ...row, decisionLog: detail.decisionLog } : detail);
      // The mutation returns a row, not history or settings. Refetch those
      // resources; never manufacture a decision or timestamp in the browser.
      await Promise.all(keys.map((queryKey) => qc.invalidateQueries({ queryKey })));
      return isCurrent();
    };
    return {
      approve: (id: string) => settle(id, "approve", () => approveUser(id)),
      reject: (id: string, reason: string) => settle(id, "reject", () => rejectUser(id, reason)),
      suspend: (id: string, reason: string) => settle(id, "suspend", () => suspendUser(id, reason)),
      unsuspend: (id: string) => settle(id, "unsuspend", () => unsuspendUser(id)),
      resetVerification: (id: string, reason: string) => settle(id, "reset verification", () => resetUserVerification(id, reason)),
    };
  }, [qc]);
}
