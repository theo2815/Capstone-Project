"use client";

import { useQuery } from "@tanstack/react-query";
import {
  searchEventByFace,
  type EventPhotosResult,
  type Photo,
} from "@/lib/api-photos";

interface UseEventFacePhotosArgs {
  slug: string;
  userId: string | undefined;
  selfieId: string | undefined;
  enabled: boolean;
}

interface UseEventFacePhotosResult {
  photos: Photo[];
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  error: unknown;
}

// Q-005 + Q-006: face search is a POST to /events/{slug}/photos/search-by-face
// (backend proxies to ai-api with event_id scoping). Cache key carries `face: userId`
// so it doesn't collide with the bib-keyed grid. Disabled when userId or selfieId
// is missing — the caller routes guests / no-primary users elsewhere first.
export function useEventFacePhotos(
  args: UseEventFacePhotosArgs,
): UseEventFacePhotosResult {
  const query = useQuery<EventPhotosResult>({
    queryKey: [
      "events",
      args.slug,
      "photos",
      { face: args.userId ?? null, selfie: args.selfieId ?? null },
    ],
    queryFn: () =>
      searchEventByFace(args.slug, { selfieId: args.selfieId! }),
    enabled: args.enabled && !!args.userId && !!args.selfieId,
    staleTime: 30_000,
  });

  return {
    photos: query.data?.items ?? [],
    total: query.data?.total ?? 0,
    isLoading: query.isPending,
    isFetching: query.isFetching,
    error: query.error,
  };
}
