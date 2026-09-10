import type { EventState } from "@/types/event";

// Event-catalog helpers. The live source for `/events` is `fetchEventsList()`
// from `lib/api-events.ts`. This module owns the lifecycle-state derivation
// (`deriveEventState`, `canUploadToEvent`, `uploadDaysRemaining`).
//
// `UPLOAD_GRACE_DAYS` is the photographer upload window (race day + 3 = 4
// days inclusive). After that the event flips to "open" (gallery for sale).
// Past `RECENT_TO_ARCHIVE_DAYS` it goes to "past" (archive).
export const UPLOAD_GRACE_DAYS = 4;
export const RECENT_TO_ARCHIVE_DAYS = 90;

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

// Photographer-owned events (V46): the owner may upload from before the event
// through one calendar month after the event date, then it closes. Mirrors the
// backend gate in PhotoUploadService.gate (OWNER_UPLOAD_GRACE_MONTHS = 1) so the
// tile + gate copy never disagree with what the server enforces. No lower bound
// — the owner set the date and may stage photos ahead of it.
export function ownerUploadCloseDate(eventDate: string): string {
  const [y, m, d] = eventDate.split("-").map(Number);
  // Clamp to end-of-month like Java's LocalDate.plusMonths(1) (Jan 31 → Feb 28)
  // so a month-end event date closes the same day the backend closes it.
  const monthIndex = m; // (m - 1) zero-based, + 1 month
  const ty = y + Math.floor(monthIndex / 12);
  const tm = monthIndex % 12;
  const lastDay = new Date(Date.UTC(ty, tm + 1, 0)).getUTCDate();
  const day = Math.min(d, lastDay);
  return new Date(Date.UTC(ty, tm, day)).toISOString().slice(0, 10);
}

// True while the owner can still push photos to their own event. Open until the
// end of the close date (one month after race day); no lower bound.
export function canOwnerUploadToEvent(
  eventDate: string,
  now: Date = new Date(),
): boolean {
  // daysSinceEvent returns (today − date) in whole days; ≤ 0 means Manila-today
  // has not passed the close date yet.
  return daysSinceEvent(ownerUploadCloseDate(eventDate), now) <= 0;
}
