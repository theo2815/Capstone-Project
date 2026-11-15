"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import {
  decideRelevanceAppeal,
  fetchRelevanceAlerts,
  fetchRelevanceAppeals,
  fetchRelevanceCase,
  fetchRelevanceCases,
  fetchRelevanceHiddenPhotos,
  hideRelevanceBatch,
  keepRelevanceCaseHidden,
  markRelevanceAlertSeen,
  markRelevanceAppealEvidenceOpened,
  markRelevanceCaseLooksFine,
  restoreRelevanceCase,
  restoreRelevancePhoto,
} from "@/lib/api-admin";
import type {
  RelevanceAlert,
  RelevanceAppealRow,
  RelevanceAppealStatus,
  RelevanceCaseAction,
  RelevanceCaseDetail,
  RelevanceCaseRow,
  RelevanceHiddenPhoto,
  RelevanceTab,
} from "@/lib/admin-relevance";

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
    // RV-3.5: the Appeals tab lists appeals, not cases.
    enabled: session.enabled && tab !== "appeals",
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

// RV-2.6: a batch's holds, restored ones included (they stay listed, greyed).
export function useRelevanceHiddenPhotos(caseId: string) {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<RelevanceHiddenPhoto & { id: string }>({
    queryKey: ["admin", "relevance", "hidden-photos", caseId, ...session.key],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchRelevanceHiddenPhotos(caseId, { offset, limit }, signal);
      // One photo can carry a hold of each kind, so the key is the pair.
      return { ...page, items: page.items.map((p) => ({ ...p, id: `${p.photoId}:${p.heldBy}` })) };
    },
    limit: 100,
    staleTime: STALE_MS,
    enabled: session.enabled && caseId !== "",
  });
}

const ACTION_CALL: Record<RelevanceCaseAction, (caseId: string) => Promise<unknown>> = {
  restore: restoreRelevanceCase,
  keep_hidden: keepRelevanceCaseHidden,
  looks_fine: markRelevanceCaseLooksFine,
  hide_batch: hideRelevanceBatch,
};

/** The admin's actions. Each resolves after the server answered and the page was read again; nothing changes before. */
export function useRelevanceActions() {
  const qc = useQueryClient();
  const refresh = () => qc.invalidateQueries({ queryKey: ["admin", "relevance"] });
  const caseAction = useMutation({
    mutationFn: ({ kind, caseId }: { kind: RelevanceCaseAction; caseId: string }) => ACTION_CALL[kind](caseId),
    onSuccess: refresh,
    // A refusal often means another admin changed the case: read it again so a retry is not the same 409.
    onError: refresh,
  });
  const photo = useMutation({
    mutationFn: ({ caseId, photoId }: { caseId: string; photoId: string }) => restoreRelevancePhoto(caseId, photoId),
    onSuccess: refresh,
    onError: refresh,
  });
  return {
    runAction: caseAction.mutateAsync,
    restorePhoto: photo.mutateAsync,
    isPending: caseAction.isPending || photo.isPending,
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

// RV-3.5: the appeals of one status, newest first. The open list's total is the count on the tab and in the rail.
export function useRelevanceAppeals(status: RelevanceAppealStatus) {
  const session = usePrivateQuerySession("ADMIN");
  return useInfiniteList<RelevanceAppealRow & { id: string }>({
    queryKey: ["admin", "relevance", "appeals", status, ...session.key],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchRelevanceAppeals({ status, offset, limit }, signal);
      return { ...page, items: page.items.map((a) => ({ ...a, id: a.appealId })) };
    },
    limit: PAGE_LIMIT,
    staleTime: STALE_MS,
    enabled: session.enabled,
  });
}

/** The open appeals' count for the tab chip and the rail: one row read for its total, under its own key. */
export function useOpenRelevanceAppealsCount(): number {
  const session = usePrivateQuerySession("ADMIN");
  const query = useQuery({
    queryKey: ["admin", "relevance", "appeals", "open-count", ...session.key],
    queryFn: ({ signal }) => fetchRelevanceAppeals({ status: "OPEN", limit: 1 }, signal),
    staleTime: STALE_MS,
    enabled: session.enabled,
  });
  return query.data?.total ?? 0;
}

/** Opening the evidence and deciding. Each resolves after the server answered; the page is read again either way. */
export function useRelevanceAppealActions() {
  const qc = useQueryClient();
  const refresh = () => qc.invalidateQueries({ queryKey: ["admin", "relevance"] });
  const evidence = useMutation({ mutationFn: (appealId: string) => markRelevanceAppealEvidenceOpened(appealId) });
  const decide = useMutation({
    mutationFn: ({ appealId, outcome, note }: { appealId: string; outcome: "WON" | "LOST"; note?: string }) =>
      decideRelevanceAppeal(appealId, note ? { outcome, note } : { outcome }),
    onSuccess: refresh,
    onError: refresh,
  });
  return {
    openEvidence: evidence.mutateAsync,
    decide: decide.mutateAsync,
    evidencePending: evidence.isPending,
    decidePending: decide.isPending,
  };
}
