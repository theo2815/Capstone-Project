"use client";

import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { captureAuthSession } from "@/lib/auth-session";
import { useAuthStore } from "@/store/auth-store";
import { ApiError } from "@/lib/api";
import * as api from "@/lib/api-photo-removal";
import { usePrivateQuerySession } from "./use-private-query-session";
import type {
  PhotoRemovalRequestDto, PhotographerRemovalRequestDto, RemovalRequestFilters, RemovalPhotoFilters, RemovalPageFilters,
  CreateRemovalRequestCommand, MatchRemovalCommand, VerifyRemovalCommand, CancelRemovalCommand, DecideRemovalCommand,
  RespondToRemovalCommand, RetryRemovalCleanupCommand, RemovalSupportUpdateCommand,
} from "@/types/photo-removal";

type SessionKey = ReturnType<typeof usePrivateQuerySession>["key"];
type RemovalRole = "ADMIN" | "PHOTOGRAPHER";
export const removalKeys = {
  all: ["photo-removal"] as const,
  admin: (session: SessionKey) => ["photo-removal", "admin", ...session] as const,
  photographer: (session: SessionKey) => ["photo-removal", "photographer", ...session] as const,
};
function scope(role: RemovalRole, session: SessionKey) {
  return role === "ADMIN" ? removalKeys.admin(session) : removalKeys.photographer(session);
}
function useRemovalQuery<T>(role: RemovalRole, suffix: readonly unknown[], read: (signal: AbortSignal) => Promise<T>, enabled = true) {
  const session = usePrivateQuerySession(role);
  return useQuery({
    queryKey: [...scope(role, session.key), ...suffix],
    enabled: session.enabled && enabled,
    queryFn: async ({ signal }) => {
      const current = useAuthStore.getState();
      if (!session.enabled || !current.isAuthenticated || current.user?.role !== role ||
        current.user.id !== session.key[0] || current.sessionVersion !== session.key[2]) throw new Error("Session changed");
      const isCurrent = captureAuthSession();
      const data = await read(signal);
      signal.throwIfAborted();
      if (!isCurrent()) throw new Error("Session changed");
      return data;
    },
  });
}
export function useAdminRemovalRequests(filters: RemovalRequestFilters = {}) {
  return useRemovalQuery("ADMIN", ["list", filters], (signal) => api.listAdminRemovalRequests(filters, signal));
}
export function useAdminRemovalRequest(id: string) {
  return useRemovalQuery("ADMIN", ["detail", id], (signal) => api.getAdminRemovalRequest(id, signal), !!id);
}
export function useRemovalPhotos(id: string, filters: RemovalPhotoFilters = {}) {
  return useRemovalQuery("ADMIN", ["photos", id, filters], (signal) => api.listRemovalPhotos(id, filters, signal), !!id);
}
export function useRemovalHistory(id: string, filters: RemovalPageFilters = {}) {
  return useRemovalQuery("ADMIN", ["history", id, filters], (signal) => api.listRemovalHistory(id, filters, signal), !!id);
}
export function useRemovalEvidence(id: string) {
  return useRemovalQuery("ADMIN", ["evidence", id], (signal) => api.getRemovalEvidence(id, signal), !!id);
}
export function usePhotographerRemovalRequests(filters: RemovalRequestFilters = {}) {
  return useRemovalQuery("PHOTOGRAPHER", ["list", filters], (signal) => api.listPhotographerRemovalRequests(filters, signal));
}
export function usePhotographerRemovalRequest(id: string) {
  return useRemovalQuery("PHOTOGRAPHER", ["detail", id], (signal) => api.getPhotographerRemovalRequest(id, signal), !!id);
}

// A conflict requires the caller to display and acknowledge the freshly read version.
function useRemovalActionContext(role: RemovalRole) {
  const qc = useQueryClient();
  const session = usePrivateQuerySession(role);
  const key = scope(role, session.key);
  const conflicts = useRef(new Set<string>());
  const pending = useRef(new Set<string>());
  const [, refresh] = useState(0);
  const sessionId = JSON.stringify(key);
  const conflictKey = (id: string) => `${sessionId}:${id}`;
  const isAllowed = () => {
    const current = useAuthStore.getState();
    return session.enabled && current.isAuthenticated && current.user?.role === role &&
      current.user.id === session.key[0] && current.sessionVersion === session.key[2];
  };
  const detailKey = (id: string) => [...key, "detail", id];
  const readDetail = (id: string, signal: AbortSignal) => role === "ADMIN"
    ? api.getAdminRemovalRequest(id, signal) : api.getPhotographerRemovalRequest(id, signal);
  async function run<T>(id: string | undefined, operation: () => Promise<T>, detailResult = true): Promise<T> {
    const isCurrent = captureAuthSession();
    if (!isAllowed()) throw new Error("Session changed");
    const commandKey = conflictKey(id ?? "create");
    if (conflicts.current.has(commandKey)) throw new Error("Review the current request before submitting a changed command.");
    if (pending.current.has(commandKey)) throw new Error("A request is already pending.");
    pending.current.add(commandKey);
    try {
      let result: T;
      try {
        result = await operation();
      } catch (error) {
        if (!isCurrent()) throw new Error("Session changed");
        if (id && error instanceof ApiError && error.status === 409) {
          conflicts.current.add(commandKey);
          refresh((version) => version + 1);
          await qc.cancelQueries({ queryKey: key });
          if (!isCurrent()) throw new Error("Session changed");
          await qc.invalidateQueries({ queryKey: key, refetchType: "none" });
          await qc.fetchQuery({ queryKey: detailKey(id), staleTime: 0, queryFn: async ({ signal }) => {
            const current = await readDetail(id, signal);
            signal.throwIfAborted();
            if (!isCurrent()) throw new Error("Session changed");
            return current;
          } }).catch(() => undefined);
          if (!isCurrent()) throw new Error("Session changed");
        }
        throw error;
      }
      if (!isCurrent()) throw new Error("Session changed");
      const affected = [key, ["events"], ["public", "events"], ["photographer", "events"], ["admin", "events"],
        ["orders"], ["me", "orders"], ["inbox"], ["admin", "kpis"]];
      await Promise.all(affected.map((queryKey) => qc.cancelQueries({ queryKey })));
      if (!isCurrent()) throw new Error("Session changed");
      if (detailResult) {
        const detail = result as PhotoRemovalRequestDto | PhotographerRemovalRequestDto;
        qc.setQueryData(detailKey(detail.id), detail);
      }
      await Promise.all(affected.map((queryKey) => qc.invalidateQueries({ queryKey })));
      if (!isCurrent()) throw new Error("Session changed");
      return result;
    } finally {
      pending.current.delete(commandKey);
    }
  }
  const reviewCurrentState = (id: string, version: number) => {
    if (!isAllowed()) return false;
    const current = qc.getQueryState<PhotoRemovalRequestDto | PhotographerRemovalRequestDto>(detailKey(id));
    if (!current?.data || current.isInvalidated || current.fetchStatus !== "idle" || current.data.version !== version) return false;
    conflicts.current.delete(conflictKey(id));
    refresh((value) => value + 1);
    return true;
  };
  return { run, mutationKey: [...key, "command"], reviewCurrentState, reviewRequired: (id: string) => conflicts.current.has(conflictKey(id)) };
}
type CaseCommand<T> = { id: string; command: T };
export function useAdminRemovalActions() {
  const context = useRemovalActionContext("ADMIN");
  const create = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: (command: CreateRemovalRequestCommand) => context.run(undefined, () => api.createRemovalRequest(command)) });
  const replaceEvidence = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: (args: { id: string; file: File; commandId: string; expectedVersion: number }) => context.run(args.id, () => api.replaceRemovalEvidence(args.id, args.file, args.commandId, args.expectedVersion)) });
  const match = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<MatchRemovalCommand>) => context.run(id, () => api.findRemovalMatches(id, command), false) });
  const verify = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<VerifyRemovalCommand>) => context.run(id, () => api.verifyRemovalRequest(id, command)) });
  const cancel = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<CancelRemovalCommand>) => context.run(id, () => api.cancelRemovalRequest(id, command)) });
  const decide = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<DecideRemovalCommand>) => context.run(id, () => api.decideRemovalRequest(id, command)) });
  const retryCleanup = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<RetryRemovalCleanupCommand>) => context.run(id, () => api.retryRemovalCleanup(id, command)) });
  const supportUpdate = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<RemovalSupportUpdateCommand>) => context.run(id, () => api.addRemovalSupportUpdate(id, command), false) });
  return { create, replaceEvidence, match, verify, cancel, decide, retryCleanup, supportUpdate, reviewCurrentState: context.reviewCurrentState, reviewRequired: context.reviewRequired };
}
export function usePhotographerRemovalActions() {
  const context = useRemovalActionContext("PHOTOGRAPHER");
  const respond = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<RespondToRemovalCommand>) => context.run(id, () => api.respondToRemovalRequest(id, command)) });
  return { respond, reviewCurrentState: context.reviewCurrentState, reviewRequired: context.reviewRequired };
}
