import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  EVENT_CATALOG,
  deriveEventState,
} from "@/lib/event-catalog";
import { MOCK_EVENT_DETAILS, getEventBySlug } from "@/app/events/mock-events";
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

// City extraction used for both mock fallback and live mapping. Real
// `Event.location` is "<venue>, <city>" — split on the last comma if present.
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
  if (BACKEND_LIVE) {
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
  return [...EVENT_CATALOG];
}

export async function fetchEventDetail(
  slug: string,
): Promise<EventDetail | null> {
  if (BACKEND_LIVE) {
    try {
      return await api.get<EventDetail>(`/events/${encodeURIComponent(slug)}`);
    } catch {
      return null;
    }
  }
  return getEventBySlug(slug);
}

export async function fetchEventSlugs(): Promise<string[]> {
  if (BACKEND_LIVE) {
    // Static params dropped per spec — events change daily, dynamic SSR is
    // canonical. This helper exists for the rare caller that still needs the
    // slug list (e.g. sitemap generation), but `/events/[slug]/page.tsx`
    // intentionally does NOT export `generateStaticParams` anymore.
    const list = await fetchEventsList({ limit: 200 });
    return list.map((e) => e.slug);
  }
  return Object.keys(MOCK_EVENT_DETAILS);
}
