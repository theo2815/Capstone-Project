"use client";

import { useInfiniteList } from "@/hooks/use-infinite-list";
import {
  fetchEventPhotos,
  type EventPhotosResult,
  type Photo,
} from "@/lib/api-photos";
import { PAGE_SIZE } from "@/lib/pagination-config";

interface UseEventPhotosArgs {
  slug: string;
  bib?: string;
  /** SSR seed: the server loader's real envelope (page 0). Gallery only — a
   *  bib filter caches independently and fetches its own first page. */
  initialPage?: EventPhotosResult;
  enabled?: boolean;
}

interface UseEventPhotosResult {
  photos: Photo[];
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
  error: unknown;
}

// Per-event photo grid (cockpit + browse share it). Real server pagination:
// Load-more advances the offset instead of slicing one capped fetch.
// Q-011: bib filter is server-side; the cache key includes the normalized bib
// so cockpit (no bib) and browse (with bib) cache — and page — independently.
export function useEventPhotos(args: UseEventPhotosArgs): UseEventPhotosResult {
  const bib = args.bib?.trim() || undefined;

  const list = useInfiniteList<Photo>({
    queryKey: ["events", args.slug, "photos", { bib: bib ?? null }],
    fetchPage: (offset, limit) =>
      fetchEventPhotos(args.slug, { bib, offset, limit }),
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    initialPage: !bib ? args.initialPage : undefined,
    enabled: args.enabled ?? true,
    staleTime: 30_000,
  });

  return {
    photos: list.items,
    total: list.total,
    isLoading: list.isLoading,
    isFetching: list.isFetching,
    hasNextPage: list.hasNextPage,
    isFetchingNextPage: list.isFetchingNextPage,
    fetchNextPage: list.fetchNextPage,
    error: list.error,
  };
}
