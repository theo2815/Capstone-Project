import { api } from "@/lib/api";
import { deriveEventState } from "@/lib/event-catalog";
import type { ListEvent } from "@/app/events/events-browser";
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

export async function fetchEventsList(
  filters: EventsListFilters = {},
): Promise<ListEvent[]> {
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
  );
  return res.items.map(eventToListEvent);
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
