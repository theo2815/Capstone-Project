"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchEventsList, fetchPublicEventsCatalog, type EventsListFilters } from "@/lib/api-events";
import type { ListEvent } from "@/types/event";

// Events change on admin edits, not by the half-minute. Cover URLs inside are
// presigned for 1 h, so 5 min is comfortably safe.
const LIST_STALE_MS = 5 * 60_000;

/** initialEvents must be a completed public catalog, never a single page. */
export function usePublicEventsCatalog(initialEvents?: ReadonlyArray<ListEvent>) {
  const query = useQuery({
    queryKey: ["public", "events", "catalog"],
    queryFn: ({ signal }) => fetchPublicEventsCatalog(signal),
    staleTime: LIST_STALE_MS,
    retry: false,
    initialData: initialEvents ? () => [...initialEvents] : undefined,
    initialDataUpdatedAt: initialEvents ? 0 : undefined,
  });
  return { data: query.data, isPending: query.isPending, isFetching: query.isFetching, error: query.error, retry: query.refetch };
}

// Shared public catalog. SSR pages can seed the cache; admin mutations
// invalidate it without copying privileged admin rows into public results.
export function usePublicEvents(
  filters: EventsListFilters = {},
  initialEvents?: ReadonlyArray<ListEvent>,
): ListEvent[] | null {
  const query = useQuery<ListEvent[]>({
    queryKey: ["public", "events", filters],
    queryFn: () => fetchEventsList(filters),
    staleTime: LIST_STALE_MS,
    initialData: initialEvents ? () => [...initialEvents] : undefined,
    initialDataUpdatedAt: initialEvents ? 0 : undefined,
  });
  return query.data ?? null;
}
