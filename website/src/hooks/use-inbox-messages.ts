"use client";

import { useInfiniteQuery, useIsMutating, useMutation, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import * as photographerApi from "@/lib/api-me-photographer-messages";
import * as runnerApi from "@/lib/api-me-runner-messages";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";

export type InboxRole = "PHOTOGRAPHER" | "RUNNER";
export const inboxKeys = {
  messages: (role: InboxRole, owner: string | null, session: number) => ["inbox", role, owner, session] as const,
};
const PAGE_SIZE = 100;
type Message = { id: string; readAt: string | null };
type Page<T> = { messages: T[]; total: number | null };
interface InboxApi<T> {
  list: (limit: number, offset: number, signal: AbortSignal) => Promise<Page<T>>;
  markRead: (id: string) => Promise<T>;
  markAllRead: () => Promise<{ markedRead: number }>;
  remove: (id: string) => Promise<{ removed: boolean }>;
}
type Action = { kind: "read" | "remove"; id: string } | { kind: "read-all" };

// Each inbox has one paginated cache shared by its bell and modal. Only the
// bell supplies a polling interval; opening a modal doesn't add another poll.
function useInboxMessages<T extends Message>(role: InboxRole, api: InboxApi<T>, enabled: boolean, pollInterval: number | false) {
  const qc = useQueryClient();
  const user = useAuthStore((s) => s.user);
  const session = useAuthStore((s) => s.sessionVersion);
  const owner = user?.id ?? null;
  const queryKey = inboxKeys.messages(role, owner, session);
  const mutationKey = [...queryKey, "action"];
  const isCurrent = () => {
    const current = useAuthStore.getState();
    return current.isAuthenticated && current.user?.role === role &&
      current.user.id === owner && current.sessionVersion === session;
  };
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam, signal }) => api.list(PAGE_SIZE, pageParam, signal),
    initialPageParam: 0,
    getNextPageParam: (last, _pages, offset) => {
      const next = offset + last.messages.length;
      if (last.messages.length === 0) return undefined;
      return (last.total === null ? last.messages.length === PAGE_SIZE : next < last.total) ? next : undefined;
    },
    enabled: enabled && user?.role === role,
    staleTime: 30_000,
    refetchInterval: pollInterval,
    refetchOnWindowFocus: "always",
  });
  const pending = useIsMutating({ mutationKey }) > 0;
  const mutation = useMutation({
    mutationKey,
    retry: false,
    mutationFn: async (action: Action): Promise<boolean> => {
      if (!isCurrent()) return false;
      let updated: T | undefined;
      try {
        if (action.kind === "read") updated = await api.markRead(action.id);
        else if (action.kind === "read-all") await api.markAllRead();
        else await api.remove(action.id);
      } catch (error) {
        if (!isCurrent()) return false;
        const message = error instanceof ApiError ? (error.errors[0]?.message ?? error.message)
          : `Couldn't ${action.kind === "remove" ? "remove notification" : "mark as read"} — please try again.`;
        useToastStore.getState().showToast({ kind: "error", message });
        throw new Error(message);
      }
      if (!isCurrent()) return false;
      await qc.cancelQueries({ queryKey, exact: true });
      if (!isCurrent()) return false;
      qc.setQueryData<InfiniteData<Page<T>, number>>(queryKey, (data) => data && ({
        ...data,
        pages: data.pages.map((page) => ({
          ...page,
          messages: action.kind === "remove" ? page.messages.filter((m) => m.id !== action.id)
            : page.messages.map((m) => updated?.id === m.id ? updated : m),
        })),
      }));
      // Re-read totals, shifted page boundaries and server-issued read-all
      // timestamps. Never invent readAt or decrement a possibly stale total.
      await qc.invalidateQueries({ queryKey, exact: true });
      return isCurrent();
    },
  });
  const perform = (action: Action) => {
    // Shared across bell/modal instances and synchronous double clicks.
    if (!isCurrent() || qc.isMutating({ mutationKey }) > 0) return Promise.resolve(false);
    return mutation.mutateAsync(action).catch(() => false);
  };
  // Offset pagination can overlap if a new row arrives between page requests.
  const messages = [...new Map(query.data?.pages.flatMap((p) => p.messages).map((m) => [m.id, m]) ?? []).values()];
  return {
    messages,
    total: query.data?.pages[0]?.total ?? null,
    hasMore: query.hasNextPage,
    loading: query.isFetching,
    loadingMore: query.isFetchingNextPage,
    error: query.error ? "Couldn't refresh your inbox. Please try again." : null,
    actionError: isCurrent() ? mutation.error?.message ?? null : null,
    pending,
    refetch: () => query.refetch(),
    loadMore: () => {
      if (!isCurrent() || qc.isFetching({ queryKey, exact: true }) > 0 || qc.isMutating({ mutationKey }) > 0) return;
      return query.fetchNextPage({ cancelRefetch: false });
    },
    markRead: (id: string) => perform({ kind: "read", id }),
    markAllRead: () => perform({ kind: "read-all" }),
    remove: (id: string) => perform({ kind: "remove", id }),
  };
}

const photographerInbox = {
  list: photographerApi.fetchMyPhotographerMessages,
  markRead: photographerApi.markMyPhotographerMessageRead,
  markAllRead: photographerApi.markAllMyPhotographerMessagesRead,
  remove: photographerApi.removeMyPhotographerMessage,
};
const runnerInbox = {
  list: runnerApi.fetchMyRunnerMessages,
  markRead: runnerApi.markMyRunnerMessageRead,
  markAllRead: runnerApi.markAllMyRunnerMessagesRead,
  remove: runnerApi.removeMyRunnerMessage,
};
export function useMyPhotographerMessages(enabled = true, pollInterval: number | false = false) {
  return useInboxMessages("PHOTOGRAPHER", photographerInbox, enabled, pollInterval);
}
export function useMyRunnerMessages(enabled = true, pollInterval: number | false = false) {
  return useInboxMessages("RUNNER", runnerInbox, enabled, pollInterval);
}
