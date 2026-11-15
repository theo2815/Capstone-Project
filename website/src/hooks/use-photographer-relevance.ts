"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { ApiError } from "@/lib/api";
import { captureAuthSession } from "@/lib/auth-session";
import {
  fetchMyRelevanceCase, fetchMyRelevancePhotos, fileMyRelevanceAppeal, setMyRelevancePhotosHidden, type PhotographerRelevanceCase,
  type RelevancePhoto,
} from "@/lib/api-photographer";
import { usePrivateQuerySession } from "./use-private-query-session";

type SessionKey = ReturnType<typeof usePrivateQuerySession>["key"];
const caseKey = (caseId: string, session: SessionKey) => ["photographer", "relevance", "case", caseId, ...session] as const;

// RV-3.4a: the photographer's own Photo Relevance case, scoped to the signed-in session.
export function usePhotographerRelevanceCase(caseId: string) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  return useQuery({
    queryKey: caseKey(caseId, session.key),
    queryFn: ({ signal }) => fetchMyRelevanceCase(caseId, signal),
    enabled: session.enabled && !!caseId,
    // A 404 is a verdict, not a blip (same rule as usePhotographerEventDetail).
    retry: (failures, error) => !(error instanceof ApiError && (error.status === 404 || error.status === 429)) && failures < 3,
  });
}

// Filing waits for the server: the page changes only to the case the server returns. A refusal re-reads the case, so
// the page shows its real state, and is thrown on with the server's message.
export function useFileRelevanceAppeal(caseId: string) {
  const qc = useQueryClient();
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const key = caseKey(caseId, session.key);
  return useMutation({
    retry: false,
    mutationFn: async (reason: string): Promise<PhotographerRelevanceCase> => {
      const isCurrent = captureAuthSession();
      try {
        const next = await fileMyRelevanceAppeal(caseId, reason);
        if (!isCurrent()) throw new Error("Session changed");
        qc.setQueryData(key, next);
        return next;
      } catch (error) {
        if (isCurrent() && error instanceof ApiError && error.status === 409) void qc.invalidateQueries({ queryKey: key });
        throw error;
      }
    },
  });
}

// RV-3.4b: the blurry photos still on sale, and the ones the photographer hid; each pages on its own.
export function usePhotographerRelevancePhotos(caseId: string, list: "blurry" | "hidden") {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  return useInfiniteList<RelevancePhoto & { id: string }>({
    queryKey: ["photographer", "relevance", list, caseId, ...session.key],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchMyRelevancePhotos(caseId, list, offset, limit, signal);
      return { ...page, items: page.items.map((p) => ({ ...p, id: p.photoId })) };
    },
    limit: 50,
    enabled: session.enabled && !!caseId,
  });
}

// The server takes at most 200 ids a request (PaginationParams.MAX_LIMIT).
const HIDE_CHUNK = 200;

// Hide or undo, in chunks of 200 one after the other; a refusal stops the rest. The grids change only after the server
// answers: both lists are read again once at the end, on a refusal too.
export function useSetRelevancePhotosHidden(caseId: string) {
  const qc = useQueryClient();
  return useMutation({
    retry: false,
    mutationFn: async ({ hide, photoIds }: { hide: boolean; photoIds: string[] }) => {
      for (let i = 0; i < photoIds.length; i += HIDE_CHUNK) {
        await setMyRelevancePhotosHidden(caseId, hide, photoIds.slice(i, i + HIDE_CHUNK));
      }
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ["photographer", "relevance"] }),
  });
}
