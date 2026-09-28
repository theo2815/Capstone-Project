"use client";

import { useRef } from "react";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { fetchPublicPhotographerEventPhotos } from "@/lib/api-photographer-public";
import {
  fetchEventPhotos,
  type EventPhotosResult,
  type Photo,
} from "@/lib/api-photos";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { useAuthStore } from "@/store/auth-store";

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
  const viewer = usePrivateQuerySession();
  const authLoading = useAuthStore((s) => s.isLoading);
  const guestSeed = !authLoading && !viewer.enabled ? args.initialPage : undefined;
  const guestPage = !bib ? guestSeed : undefined;
  const scope = JSON.stringify([args.slug, ...viewer.key, authLoading]);
  const snapshotRef = useRef({ scope, value: guestSeed?.snapshotAt });
  if (snapshotRef.current.scope !== scope) {
    snapshotRef.current = { scope, value: guestSeed?.snapshotAt };
  }

  const list = useInfiniteList<Photo>({
    queryKey: [
      "events",
      args.slug,
      "photos",
      { bib: bib ?? null, snapshotAt: guestPage?.snapshotAt ?? null },
      ...viewer.key,
      authLoading,
    ],
    fetchPage: async (offset, limit, signal) => {
      const page = await fetchEventPhotos(args.slug, {
        bib,
        offset,
        limit,
        // Page zero starts a new gallery snapshot; later pages reuse it.
        snapshotAt: !bib && offset > 0 ? snapshotRef.current.value : undefined,
      }, signal);
      if (!bib && page.snapshotAt && snapshotRef.current.scope === scope) {
        snapshotRef.current.value = page.snapshotAt;
      }
      return page;
    },
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    initialPage: guestPage,
    enabled: (args.enabled ?? true) && !authLoading,
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

export function usePublicPhotographerEventPhotos(handle: string, slug: string) {
  const viewer = usePrivateQuerySession();
  return useInfiniteList<Photo>({
    queryKey: ["photographer", "public", handle, slug, "photos", ...viewer.key],
    fetchPage: (offset, limit, signal) =>
      fetchPublicPhotographerEventPhotos(handle, slug, { offset, limit }, signal),
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    staleTime: 60_000,
  });
}
