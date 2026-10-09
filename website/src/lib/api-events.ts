import { api } from "@/lib/api";
import { deriveEventState } from "@/lib/event-catalog";
import type { ListEvent } from "@/types/event";
import type { Event, EventDetail } from "@/types/event";
import type { PaginatedResponse } from "@/types/api";

export interface EventsListFilters {
  search?: string;
  city?: string;
  dateFrom?: string;
  dateTo?: string;
  offset?: number;
  limit?: number;
}

function extractCity(location: string): string {
  const idx = location.lastIndexOf(",");
  return idx === -1 ? location.trim() : location.slice(idx + 1).trim();
}

function eventToListEvent(e: Event): ListEvent {
  return {
    ...e,
    city: extractCity(e.location),
    state: deriveEventState(e.date),
  };
}

export async function fetchEventsPage(
  filters: EventsListFilters = {},
  signal?: AbortSignal,
): Promise<PaginatedResponse<ListEvent>> {
  const params = new URLSearchParams();
  // DRAFT is filtered out server-side (Q-001) — runners never see drafts.
  params.set("status", "ACTIVE,COMPLETED,ARCHIVED");
  if (filters.search) params.set("search", filters.search);
  if (filters.city) params.set("city", filters.city);
  if (filters.dateFrom) params.set("dateFrom", filters.dateFrom);
  if (filters.dateTo) params.set("dateTo", filters.dateTo);
  params.set("offset", String(filters.offset ?? 0));
  params.set("limit", String(filters.limit ?? 200));
  const res = await api.get<PaginatedResponse<Event>>(
    `/events?${params.toString()}`,
    { signal },
  );
  return { ...res, items: res.items.map(eventToListEvent) };
}

export async function fetchEventsList(
  filters: EventsListFilters = {},
): Promise<ListEvent[]> {
  return (await fetchEventsPage(filters)).items;
}

// ponytail: full-catalog cost grows with events; move facets AND results to server pagination when measured payload/latency warrants it.
export async function fetchPublicEventsCatalog(signal?: AbortSignal): Promise<ListEvent[]> {
  const incomplete = () => new Error("Events changed or could not be fully loaded. Please refresh and try again.");
  for (let attempt = 0; attempt < 2; attempt++) {
    const events = new Map<string, ListEvent>();
    let offset = 0;
    let total: number | undefined;
    while (true) {
      signal?.throwIfAborted();
      const page = await fetchEventsPage({ offset }, signal);
      signal?.throwIfAborted();
      if (page.offset !== offset || !Number.isSafeInteger(page.total) || page.total < 0) throw incomplete();
      if (total !== undefined && page.total !== total) break;
      total = page.total;
      if (page.items.length === 0 && offset < total) throw incomplete();
      let overlap = false;
      for (const event of page.items) {
        if (events.has(event.id)) overlap = true;
        events.set(event.id, event);
      }
      offset += page.items.length;
      if (overlap || offset > total) break;
      if (offset === total) return [...events.values()];
    }
  }
  throw incomplete();
}

export async function fetchEventDetail(
  slug: string,
): Promise<EventDetail | null> {
  try {
    return await api.get<EventDetail>(`/events/${encodeURIComponent(slug)}`);
  } catch {
    return null;
  }
}

export async function fetchEventSlugs(): Promise<string[]> {
  const list = await fetchEventsList({ limit: 200 });
  return list.map((e) => e.slug);
}
