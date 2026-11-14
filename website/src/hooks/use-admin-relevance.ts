"use client";

import { useQuery } from "@tanstack/react-query";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { fetchRelevanceCase, fetchRelevanceCases } from "@/lib/api-admin";
import type { RelevanceCaseDetail, RelevanceCaseRow, RelevanceTab } from "@/lib/admin-relevance";

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

export function useRelevanceCase(caseId: string) {
  const session = usePrivateQuerySession("ADMIN");
  return useQuery<RelevanceCaseDetail>({
    queryKey: ["admin", "relevance", "case", caseId, ...session.key],
    queryFn: ({ signal }) => fetchRelevanceCase(caseId, signal),
    staleTime: STALE_MS,
    enabled: session.enabled && caseId !== "",
  });
}
