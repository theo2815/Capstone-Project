"use client";

import { useIsMutating, useMutation, useQueryClient, type Mutation } from "@tanstack/react-query";
import { usePrivateQuerySession } from "./use-private-query-session";
import { useAuthStore } from "@/store/auth-store";

export type AdminTarget = { kind: "user" | "event" | "flag" | "dispute" | "payout" | "report" | "security"; id: string };
export type AdminMutationSpec<T> = { targets: readonly AdminTarget[]; action: string; execute: () => Promise<T> };
type Result<T> = { kind: "completed"; value: T } | { kind: "blocked" | "stale" };
type Command = AdminMutationSpec<unknown> & { session: ReturnType<typeof usePrivateQuerySession>["key"] };
const mutationKey = ["admin", "action"] as const;

function isCurrent(session: Command["session"]) {
  const current = useAuthStore.getState();
  return current.isAuthenticated && current.user?.role === "ADMIN" &&
    current.user.id === session[0] && session[1] === "ADMIN" && current.sessionVersion === session[2];
}

export function useAdminMutation() {
  const qc = useQueryClient();
  const session = usePrivateQuerySession("ADMIN");
  const matches = (mutation: Mutation, targets: readonly AdminTarget[]) => {
    const command = mutation.state.variables as Command;
    return isCurrent(session.key) && command.session.every((value, index) => value === session.key[index]) &&
      command.targets.some((target) => targets.some((candidate) => candidate.kind === target.kind && candidate.id === target.id));
  };
  useIsMutating({ mutationKey });
  const mutation = useMutation({
    mutationKey,
    retry: false,
    mutationFn: async (command: Command): Promise<Result<unknown>> => {
      if (!isCurrent(command.session)) return { kind: "stale" };
      try {
        const value = await command.execute();
        return isCurrent(command.session) ? { kind: "completed", value } : { kind: "stale" };
      } catch (error) {
        if (!isCurrent(command.session)) return { kind: "stale" };
        throw error;
      }
    },
  });
  const isPending = (targets: readonly AdminTarget[]) => qc.isMutating({ mutationKey, predicate: (mutation) => matches(mutation, targets) }) > 0;
  return {
    run: <T,>(spec: AdminMutationSpec<T>): Promise<Result<T>> => {
      if (!isCurrent(session.key)) return Promise.resolve({ kind: "stale" });
      if (isPending(spec.targets)) return Promise.resolve({ kind: "blocked" });
      return mutation.mutateAsync({ ...spec, targets: spec.targets.map((target) => ({ ...target })), session: session.key }) as Promise<Result<T>>;
    },
    isPending,
    pendingAction: (targets: readonly AdminTarget[]): string | null => {
      const pending = qc.getMutationCache().find({ mutationKey, status: "pending", predicate: (mutation) => matches(mutation, targets) });
      return pending ? (pending.state.variables as Command).action : null;
    },
  };
}
