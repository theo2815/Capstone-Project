"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchEventsList, type EventsListFilters } from "@/lib/api-events";
import type { ListEvent } from "@/app/events/events-browser";

// Events change on admin edits, not by the half-minute. Cover URLs inside are
// presigned for 1 h, so 5 min is comfortably safe.
const LIST_STALE_MS = 5 * 60_000;

// Client-side fetcher for the public events catalog. /events SSR-loads this
// list, but dashboard surfaces (e.g. /dashboard/upload) can't SSR because the
// page is interactive, so they read through here. Pass the result as the
// seed to useEventCatalog() to keep admin overrides + tombstones merged in.
export function usePublicEvents(
  filters: EventsListFilters = {},
): ListEvent[] | null {
  const query = useQuery<ListEvent[]>({
    queryKey: ["public", "events", filters],
    queryFn: () => fetchEventsList(filters),
    staleTime: LIST_STALE_MS,
  });
  return query.data ?? null;
}
