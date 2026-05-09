import { useMemo } from "react";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";

// Mock catalog. Lifted out of `app/events/page.tsx` so /profile's Race Log can
// resolve event metadata by ID without re-importing client modules.
//
// Used as the mock-fallback source for `lib/api-events.ts` when
// BACKEND_LIVE=false. When live, `fetchEventsList()` returns server data
// instead. Race Log derives from saved-events ∪ orders and resolves names via
// `getEventById()` here today; future Phase E may add a server-side join.

// Lifecycle is now derived from `date` instead of carried in the seed.
// `UPLOAD_GRACE_DAYS` is the photographer upload window (race day + 3 = 4
// days inclusive). After that the event flips to "open" (gallery for sale).
// Past `RECENT_TO_ARCHIVE_DAYS` it goes to "past" (archive).
export const UPLOAD_GRACE_DAYS = 4;
export const RECENT_TO_ARCHIVE_DAYS = 90;

type SeedEvent = Omit<ListEvent, "state">;

const SEED: ReadonlyArray<SeedEvent> = [
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
  },
];

// Whole-day delta from event date to today. Negative = future, 0 = race day,
// positive = past. Both sides anchored to local midnight so the boundaries
// flip cleanly at 00:00 instead of mid-day.
function daysSinceEvent(eventDate: string, now: Date = new Date()): number {
  const event = new Date(`${eventDate}T00:00:00`);
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const ms = today.getTime() - event.getTime();
  return Math.floor(ms / (1000 * 60 * 60 * 24));
}

export function deriveEventState(
  eventDate: string,
  now: Date = new Date(),
): EventState {
  const days = daysSinceEvent(eventDate, now);
  if (days < 0) return "upcoming";
  if (days < UPLOAD_GRACE_DAYS) return "live";
  if (days < RECENT_TO_ARCHIVE_DAYS) return "open";
  return "past";
}

// True when photographer can still push photos to this event. Race day
// (day 0) through day 3 inclusive — 4 days total. Day 4 onward is closed.
export function canUploadToEvent(
  eventDate: string,
  now: Date = new Date(),
): boolean {
  const days = daysSinceEvent(eventDate, now);
  return days >= 0 && days < UPLOAD_GRACE_DAYS;
}

// Days remaining in the upload window (0 = closes at end of today, null = not
// open yet or already closed). Used by the upload tile to nudge "1 day left."
export function uploadDaysRemaining(
  eventDate: string,
  now: Date = new Date(),
): number | null {
  const days = daysSinceEvent(eventDate, now);
  if (days < 0 || days >= UPLOAD_GRACE_DAYS) return null;
  return UPLOAD_GRACE_DAYS - 1 - days;
}

// Public catalog seed — readers can reference but we recompute state on read.
export const EVENT_CATALOG: ReadonlyArray<ListEvent> = SEED.map((e) => ({
  ...e,
  state: deriveEventState(e.date),
}));

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
  const store = useAdminEventOverridesStore.getState();
  return mergeAdminCatalog(seed, store.overrides, store.submissions, store.tombstones);
}

// Reactive hook. Subscribes to the admin override store so the consuming
// component re-renders when admin creates/edits/deletes. Pass a custom `seed`
// only if the page received its own (e.g. the server-rendered /events page).
export function useEventCatalog(
  seed: ReadonlyArray<ListEvent> = EVENT_CATALOG,
): ReadonlyArray<ListEvent> {
  const overrides = useAdminEventOverridesStore((s) => s.overrides);
  const submissions = useAdminEventOverridesStore((s) => s.submissions);
  const tombstones = useAdminEventOverridesStore((s) => s.tombstones);
  return useMemo(
    () => mergeAdminCatalog(seed, overrides, submissions, tombstones),
    [seed, overrides, submissions, tombstones],
  );
}

function mergeAdminCatalog(
  seed: ReadonlyArray<ListEvent>,
  overrides: Record<string, Partial<ListEvent>>,
  submissions: ReadonlyArray<ListEvent>,
  tombstones: ReadonlyArray<string>,
): ReadonlyArray<ListEvent> {
  const tombstoned = new Set(tombstones);
  const patched = seed
    .filter((e) => !tombstoned.has(e.id))
    .map((e) => {
      const patch = overrides[e.id];
      if (!patch) return e;
      const merged = { ...e, ...patch };
      // Derived state always wins over any stale `state` patch the store
      // may carry (legacy `setEventState` calls). Date can change via edit;
      // recompute from the merged date.
      merged.state = deriveEventState(merged.date);
      return merged;
    });
  // Submissions go in front so they sort naturally with the rest.
  return [...submissions, ...patched];
}
