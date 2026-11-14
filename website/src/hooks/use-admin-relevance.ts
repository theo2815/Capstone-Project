"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import {
  fetchRelevanceAlerts,
  fetchRelevanceCase,
  fetchRelevanceCases,
  markRelevanceAlertSeen,
} from "@/lib/api-admin";
import type { RelevanceAlert, RelevanceCaseDetail, RelevanceCaseRow, RelevanceTab } from "@/lib/admin-relevance";

// RV-1.11a: every key sits under ["admin", "relevance"], so one invalidation refreshes the page.
const STALE_MS = 60_000;
const PAGE_LIMIT = 50;

export function useRelevanceCases(tab: RelevanceTab) {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<RelevanceCaseRow & { id: string }>({
    queryKey: ["admin", "relevance", "cases", tab, ...session.key],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchRelevanceCases({ tab, offset, limit }, signal);
      return { ...page, items: page.items.map((c) => ({ ...c, id: c.caseId })) };
    },
    limit: PAGE_LIMIT,
    staleTime: STALE_MS,
    enabled: session.enabled,
  });
}

// RV-1.11b: the unseen alerts, shared by the inbox chip (its count) and the inbox list.
export function useUnseenRelevanceAlerts() {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<RelevanceAlert & { id: string }>({
    queryKey: ["admin", "relevance", "alerts", "unseen", ...session.key],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchRelevanceAlerts({ unseen: true, offset, limit }, signal);
      return { ...page, items: page.items.map((a) => ({ ...a, id: a.alertId })) };
    },
    limit: PAGE_LIMIT,
    staleTime: STALE_MS,
    enabled: session.enabled,
  });
}

/** Marks one alert seen; the list changes only after the server confirms and the alerts are read again. */
export function useMarkRelevanceAlertSeen() {
  const qc = useQueryClient();
  const mutation = useMutation({
    mutationFn: (alertId: string) => markRelevanceAlertSeen(alertId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["admin", "relevance"] }),
  });
  return {
    markSeen: mutation.mutateAsync,
    isPending: (alertId: string) => mutation.isPending && mutation.variables === alertId,
  };
}

export function useRelevanceCase(caseId: string) {
  const session = usePrivateQuerySession("ADMIN");
  return useQuery<RelevanceCaseDetail>({
    queryKey: ["admin", "relevance", "case", caseId, ...session.key],
    queryFn: ({ signal }) => fetchRelevanceCase(caseId, signal),
    staleTime: STALE_MS,
    enabled: session.enabled && caseId !== "",
  });
}
