import { useMemo } from "react";
import type { ListEvent } from "@/app/events/events-browser";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";

// Mock catalog. Lifted out of `app/events/page.tsx` so /profile's Race Log can
// resolve event metadata by ID without re-importing client modules.
//
// TODO(backend): replace with `api.get<Event[]>("/events?...")` paginated; the
// race log will eventually fetch only the events the user has touched.

export const EVENT_CATALOG: ReadonlyArray<ListEvent> = [
  {
    id: "u1",
    slug: "cebu-bay-run-2026",
    name: "Cebu Bay Run",
    date: "2026-05-09",
    location: "Mactan Channel Bridge",
    city: "Mactan",
    photoCount: 0,
    participantCount: 1500,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "u2",
    slug: "it-park-sunrise-5k-2026",
    name: "IT Park Sunrise 5K",
    date: "2026-05-17",
    location: "IT Park, Cebu City",
    city: "Cebu City",
    photoCount: 0,
    participantCount: 800,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "u3",
    slug: "cordova-foundation-run-2026",
    name: "Cordova Foundation Run",
    date: "2026-06-07",
    location: "Cordova, Cebu",
    city: "Cordova",
    photoCount: 0,
    participantCount: 600,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "1",
    slug: "cebu-marathon-2026",
    name: "Cebu Marathon 2026",
    date: "2026-04-28",
    location: "SRP Boulevard, Cebu City",
    city: "Cebu City",
    photoCount: 1240,
    participantCount: 4800,
    status: "ACTIVE",
    state: "live",
  },
  {
    id: "2",
    slug: "mactan-sunset-run-2026",
    name: "Mactan Sunset Run",
    date: "2026-04-26",
    location: "Mactan Channel Bridge",
    city: "Mactan",
    photoCount: 612,
    participantCount: 1800,
    status: "ACTIVE",
    state: "live",
  },
  {
    id: "3",
    slug: "srp-half-marathon-2026",
    name: "SRP Half-Marathon",
    date: "2026-04-12",
    location: "South Road Properties, Cebu",
    city: "Cebu City",
    photoCount: 3850,
    participantCount: 2400,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "4",
    slug: "sun-run-cebu-2026",
    name: "Sun Run Cebu",
    date: "2026-04-05",
    location: "IT Park, Cebu City",
    city: "Cebu City",
    photoCount: 2120,
    participantCount: 3100,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "5",
    slug: "mactan-coastal-5k-2026",
    name: "Mactan Coastal 5K",
    date: "2026-03-29",
    location: "Mactan, Lapu-Lapu City",
    city: "Lapu-Lapu",
    photoCount: 980,
    participantCount: 1200,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "6",
    slug: "cebu-night-run-2025",
    name: "Cebu City Night Run 2025",
    date: "2025-12-14",
    location: "Cebu Business Park",
    city: "Cebu City",
    photoCount: 4200,
    participantCount: 5600,
    status: "ARCHIVED",
    state: "past",
  },
  {
    id: "7",
    slug: "talisay-10k-2025",
    name: "Talisay 10K",
    date: "2025-11-09",
    location: "Talisay City, Cebu",
    city: "Talisay",
    photoCount: 1640,
    participantCount: 1900,
    status: "ARCHIVED",
    state: "past",
  },
  {
    id: "8",
    slug: "cordova-bayrun-2025",
    name: "Cordova Bay Run",
    date: "2025-10-19",
    location: "Cordova, Cebu",
    city: "Cordova",
    photoCount: 720,
    participantCount: 950,
    status: "ARCHIVED",
    state: "past",
  },
];

// Mocked per-event "photos found of you" counts for the demo Race Log.
// Real value comes from ai-api face-search results scoped to the event +
// the user's selfie embeddings (backend Phase D).
export const MOCK_USER_PHOTOS_FOUND: Record<string, number> = {
  "1": 12,
  "3": 8,
  "6": 4,
};

export function getEventById(id: string): ListEvent | undefined {
  return getCatalogWithOverrides().find((e) => e.id === id);
}

// Non-reactive merged view. Use inside event handlers, server components,
// or any code that reads the catalog once. For reactive consumers (UI that
// must re-render when admin flips a state), use `useEventCatalog()` below.
export function getCatalogWithOverrides(
  seed: ReadonlyArray<ListEvent> = EVENT_CATALOG,
): ReadonlyArray<ListEvent> {
  const overrides = useAdminEventOverridesStore.getState().overrides;
  if (Object.keys(overrides).length === 0) return seed;
  return seed.map((e) => {
    const patch = overrides[e.id];
    return patch ? { ...e, ...patch } : e;
  });
}

// Reactive hook. Subscribes to the admin override store so the consuming
// component re-renders when admin flips a state. Pass a custom `seed` only
// if the page received its own (e.g. the server-rendered /events page that
// hydrates EventsBrowser with the canonical seed).
export function useEventCatalog(
  seed: ReadonlyArray<ListEvent> = EVENT_CATALOG,
): ReadonlyArray<ListEvent> {
  const overrides = useAdminEventOverridesStore((s) => s.overrides);
  return useMemo(() => {
    if (Object.keys(overrides).length === 0) return seed;
    return seed.map((e) => {
      const patch = overrides[e.id];
      return patch ? { ...e, ...patch } : e;
    });
  }, [seed, overrides]);
}
