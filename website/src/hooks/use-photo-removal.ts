"use client";

import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { captureAuthSession } from "@/lib/auth-session";
import { useAuthStore } from "@/store/auth-store";
import { ApiError } from "@/lib/api";
import * as api from "@/lib/api-photo-removal";
import { usePrivateQuerySession } from "./use-private-query-session";
import { useInfiniteList } from "./use-infinite-list";
import type {
  PhotoRemovalRequestDto, PhotographerRemovalRequestDto, RemovalRequestFilters, RemovalPhotoFilters, RemovalPageFilters, RemovalPage,
  CreateRemovalRequestCommand, MatchRemovalCommand, VerifyRemovalCommand, CancelRemovalCommand, DecideRemovalCommand,
  RespondToRemovalCommand, RetryRemovalCleanupCommand, RemovalSupportUpdateCommand, ReviewSuggestionsCommand, RunnerRemovalRequestDetailDto,
} from "@/types/photo-removal";
type RemovalDetail = PhotoRemovalRequestDto | PhotographerRemovalRequestDto | RunnerRemovalRequestDetailDto;

type SessionKey = ReturnType<typeof usePrivateQuerySession>["key"];
type RemovalRole = "ADMIN" | "PHOTOGRAPHER" | "RUNNER";
export const removalKeys = {
  all: ["photo-removal"] as const,
  admin: (session: SessionKey) => ["photo-removal", "admin", ...session] as const,
  photographer: (session: SessionKey) => ["photo-removal", "photographer", ...session] as const,
  runner: (session: SessionKey) => ["photo-removal", "runner", ...session] as const,
};
function scope(role: RemovalRole, session: SessionKey) {
  if (role === "RUNNER") return removalKeys.runner(session);
  return role === "ADMIN" ? removalKeys.admin(session) : removalKeys.photographer(session);
}
// A read that started under one session never lands in another session's cache.
async function readForSession<T>(role: RemovalRole, session: ReturnType<typeof usePrivateQuerySession>, signal: AbortSignal, read: (signal: AbortSignal) => Promise<T>) {
  const current = useAuthStore.getState();
  if (!session.enabled || !current.isAuthenticated || current.user?.role !== role ||
    current.user.id !== session.key[0] || current.sessionVersion !== session.key[2]) throw new Error("Session changed");
  const isCurrent = captureAuthSession();
  const data = await read(signal);
  signal.throwIfAborted();
  if (!isCurrent()) throw new Error("Session changed");
  return data;
}
function useRemovalQuery<T>(role: RemovalRole, suffix: readonly unknown[], read: (signal: AbortSignal) => Promise<T>, enabled = true) {
  const session = usePrivateQuerySession(role);
  return useQuery({
    queryKey: [...scope(role, session.key), ...suffix],
    enabled: session.enabled && enabled,
    queryFn: ({ signal }) => readForSession(role, session, signal, read),
  });
}
// AX-36: admin lists load more pages instead of Prev/Next. Commands only invalidate this scope, never patch it.
function useAdminRemovalList<T extends { id: string }>(suffix: readonly unknown[], limit: number,
  read: (page: { offset: number; limit: number }, signal: AbortSignal) => Promise<RemovalPage<T>>, enabled = true) {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<T>({
    queryKey: [...scope("ADMIN", session.key), ...suffix],
    enabled: session.enabled && enabled,
    limit,
    fetchPage: (offset, pageLimit, signal) => readForSession("ADMIN", session, signal, (s) => read({ offset, limit: pageLimit }, s)),
  });
}
export function useAdminRemovalRequests(filters: Omit<RemovalRequestFilters, "offset" | "limit"> = {}, limit = 20) {
  return useAdminRemovalList(["list", filters], limit, (page, signal) => api.listAdminRemovalRequests({ ...filters, ...page }, signal));
}
export function useAdminRemovalRequest(id: string) {
  return useRemovalQuery("ADMIN", ["detail", id], (signal) => api.getAdminRemovalRequest(id, signal), !!id);
}
// Gallery rows carry no top-level id; the photo id keys them.
export function useRemovalPhotos(id: string, filters: Omit<RemovalPhotoFilters, "offset" | "limit"> = {}, limit = 24) {
  return useAdminRemovalList(["photos", id, filters], limit, async (page, signal) => {
    const result = await api.listRemovalPhotos(id, { ...filters, ...page }, signal);
    return { ...result, items: result.items.map((item) => ({ ...item, id: item.photo.id })) };
  }, !!id);
}
export function useRemovalHistory(id: string, limit = 50) {
  return useAdminRemovalList(["history", id], limit, (page, signal) => api.listRemovalHistory(id, page, signal), !!id);
}
export function useRemovalEvidence(id: string) {
  return useRemovalQuery("ADMIN", ["evidence", id], (signal) => api.getRemovalEvidence(id, signal), !!id);
}
export function useRunnerRemovalReasons() {
  return useRemovalQuery("RUNNER", ["reasons"], (signal) => api.listRunnerRemovalReasons(signal));
}
export function useRunnerRemovalEligibility(eventId: string | null) {
  return useRemovalQuery("RUNNER", ["eligibility", eventId], (signal) => api.getRunnerRemovalEligibility(eventId!, signal), !!eventId);
}
export function useRunnerRemovalRequests(filters: RemovalPageFilters = {}) {
  return useRemovalQuery("RUNNER", ["list", filters], (signal) => api.listRunnerRemovalRequests(filters, signal));
}
export function useRunnerRemovalRequest(id: string) {
  return useRemovalQuery("RUNNER", ["detail", id], (signal) => api.getRunnerRemovalRequest(id, signal), !!id);
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
  const readDetail = (id: string, signal: AbortSignal): Promise<RemovalDetail> => role === "RUNNER"
    ? api.getRunnerRemovalRequest(id, signal)
    : role === "ADMIN" ? api.getAdminRemovalRequest(id, signal) : api.getPhotographerRemovalRequest(id, signal);
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
      const affected = [key, ["events"], ["public", "events"], ["photographer", "events"], ["photographer", "public"], ["photographer", "photos"], ["admin", "events"],
        ["orders"], ["me", "orders"], ["inbox"], ["admin", "kpis"]];
      await Promise.all(affected.map((queryKey) => qc.cancelQueries({ queryKey })));
      if (!isCurrent()) throw new Error("Session changed");
      if (detailResult) {
        const detail = result as RemovalDetail;
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
    const current = qc.getQueryState<RemovalDetail>(detailKey(id));
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
  const reviewSuggestions = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<ReviewSuggestionsCommand>) => context.run(id, () => api.reviewRemovalSuggestions(id, command)) });
  return { create, replaceEvidence, match, verify, cancel, decide, retryCleanup, supportUpdate, reviewSuggestions, reviewCurrentState: context.reviewCurrentState, reviewRequired: context.reviewRequired };
}
export function usePhotographerRemovalActions() {
  const context = useRemovalActionContext("PHOTOGRAPHER");
  const respond = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, command }: CaseCommand<RespondToRemovalCommand>) => context.run(id, () => api.respondToRemovalRequest(id, command)) });
  return { respond, reviewCurrentState: context.reviewCurrentState, reviewRequired: context.reviewRequired };
}
export function useRunnerRemovalActions() {
  const context = useRemovalActionContext("RUNNER");
  const withdraw = useMutation({ mutationKey: context.mutationKey, retry: false, mutationFn: ({ id, expectedVersion, idempotencyKey }: { id: string; expectedVersion: number; idempotencyKey: string }) => context.run(id, () => api.withdrawRunnerRemoval(id, expectedVersion, idempotencyKey)) });
  return { withdraw, reviewCurrentState: context.reviewCurrentState, reviewRequired: context.reviewRequired };
}
