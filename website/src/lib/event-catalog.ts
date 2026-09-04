import { useMemo } from "react";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";

// Event-catalog helpers. The live source for `/events` is `fetchEventsList()`
// from `lib/api-events.ts`. This module owns the lifecycle-state derivation
// (`deriveEventState`, `canUploadToEvent`, `uploadDaysRemaining`) and the
// admin-override merge (`useEventCatalog` / `getCatalogWithOverrides`).
//
// `UPLOAD_GRACE_DAYS` is the photographer upload window (race day + 3 = 4
// days inclusive). After that the event flips to "open" (gallery for sale).
// Past `RECENT_TO_ARCHIVE_DAYS` it goes to "past" (archive).
export const UPLOAD_GRACE_DAYS = 4;
export const RECENT_TO_ARCHIVE_DAYS = 90;

type SeedEvent = Omit<ListEvent, "state">;

const SEED: ReadonlyArray<SeedEvent> = [];

// Every race is in the Philippines, so "today" is the Manila calendar date
// wherever this runs — Vercel lambdas are UTC and reserve the TZ env var, and
// a browser can be anywhere. Without this, the server labelled a race
// "upcoming" until 08:00 PHT while the runner's browser said "live".
const EVENT_TZ = "Asia/Manila";
const manilaDate = new Intl.DateTimeFormat("en-CA", { timeZone: EVENT_TZ }); // YYYY-MM-DD

// Whole-day delta from event date to today. Negative = future, 0 = race day,
// positive = past. Both dates parse as UTC midnight, so the diff is exact.
function daysSinceEvent(eventDate: string, now: Date = new Date()): number {
  const today = manilaDate.format(now);
  return Math.round((Date.parse(today) - Date.parse(eventDate)) / 86_400_000);
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

// Per-event "photos found of you" counts. Real values come from ai-api
// face-search results scoped to the event + the user's selfie embeddings.
// Empty until that hook lands.
export const MOCK_USER_PHOTOS_FOUND: Record<string, number> = {};

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
  // Once the post-create refetch surfaces the row in `seed`, the local
  // optimistic submission becomes a duplicate. Drop submissions whose id
  // already appears in seed so React sees one card per event-id and the
  // "two children with the same key" warning stays quiet.
  const seedIds = new Set(patched.map((e) => e.id));
  const optimistic = submissions.filter((e) => !seedIds.has(e.id));
  // Submissions go in front so they sort naturally with the rest.
  return [...optimistic, ...patched];
}
