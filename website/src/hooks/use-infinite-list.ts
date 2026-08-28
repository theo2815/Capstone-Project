"use client";

import { useInfiniteQuery, type QueryKey } from "@tanstack/react-query";
import type { PaginatedResponse } from "@/types/api";

// Shared offset/limit pagination adapter over React Query's useInfiniteQuery.
// The backend's list contract is uniform — every paginated endpoint returns
// PaginatedResponse<T> = { items, total, offset, limit } and pages by absolute
// row offset (next page = offset + limit). This encodes that math once so the
// ~12 surface hooks don't each re-derive getNextPageParam / flatten / total.
//
// SSR seed: pass `initialPage` (the server loader's real envelope) and it maps
// to page 0 with the real `total` — no more `total = items.length` poisoning.

interface UseInfiniteListArgs<T> {
  queryKey: QueryKey;
  fetchPage: (offset: number, limit: number) => Promise<PaginatedResponse<T>>;
  limit: number;
  initialPage?: PaginatedResponse<T>;
  enabled?: boolean;
  staleTime?: number;
}

interface UseInfiniteListResult<T> {
  items: T[];
  total: number;
  isLoading: boolean;
  isFetching: boolean;
  isFetchingNextPage: boolean;
  hasNextPage: boolean;
  fetchNextPage: () => void;
  error: unknown;
}

export function useInfiniteList<T>({
  queryKey,
  fetchPage,
  limit,
  initialPage,
  enabled,
  staleTime,
}: UseInfiniteListArgs<T>): UseInfiniteListResult<T> {
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam }) => fetchPage(pageParam, limit),
    initialPageParam: 0,
    getNextPageParam: (last) => {
      const next = last.offset + last.limit;
      return next < last.total ? next : undefined;
    },
    enabled: enabled ?? true,
    initialData: initialPage
      ? { pages: [initialPage], pageParams: [0] }
      : undefined,
    staleTime,
  });

  const pages = query.data?.pages ?? [];
  return {
    items: pages.flatMap((p) => p.items),
    // Every page carries the same server total; read the latest fetched one.
    total: pages.at(-1)?.total ?? 0,
    isLoading: query.isPending,
    isFetching: query.isFetching,
    isFetchingNextPage: query.isFetchingNextPage,
    hasNextPage: query.hasNextPage,
    fetchNextPage: () => void query.fetchNextPage(),
    error: query.error,
  };
}
