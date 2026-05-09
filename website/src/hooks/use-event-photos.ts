"use client";

import { useQuery } from "@tanstack/react-query";
import {
  fetchEventPhotos,
  type EventPhotosResult,
  type Photo,
} from "@/lib/api-photos";

interface UseEventPhotosArgs {
  slug: string;
  bib?: string;
  initialItems?: Photo[];
  enabled?: boolean;
}

interface UseEventPhotosResult {
  photos: Photo[];
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  error: unknown;
}

// Hook for the per-event photo grid (cockpit + browse modes share it).
// Q-011: bib filter is server-side; cache key includes the normalized bib so
// cockpit (no bib) and browse (with bib) cache independently.
export function useEventPhotos(args: UseEventPhotosArgs): UseEventPhotosResult {
  const bib = args.bib?.trim() || undefined;

  const query = useQuery<EventPhotosResult>({
    queryKey: ["events", args.slug, "photos", { bib: bib ?? null }],
    queryFn: () => fetchEventPhotos(args.slug, { bib }),
    enabled: args.enabled ?? true,
    initialData:
      !bib && args.initialItems
        ? {
            items: args.initialItems,
            total: args.initialItems.length,
            offset: 0,
            limit: args.initialItems.length,
          }
        : undefined,
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
