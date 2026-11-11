import type { EventState, ListEvent } from "@/types/event";
import type {
  PendingPricingChange,
  PhotographerEventSummary,
} from "@/lib/photographer-mock";
import type { RaceCreateResult } from "@/lib/api-photographer";
import { formatPrice } from "@/lib/utils";

// Photographer-owned events (V46): the few derivations the dashboard pages,
// the upload picker and the admin queue all need, so they agree on wording.

// Build the ListEvent shape EventTile expects from a BE summary. The BE row
// already carries every field the dashboard tiles read; participantCount +
// status are slot fillers, city is derived from "Venue, City".
export function summaryToListEvent(p: PhotographerEventSummary): ListEvent {
  return {
    id: p.id,
    slug: p.slug,
    name: p.name,
    date: p.date,
    location: p.location,
    bannerUrl: p.bannerUrl ?? undefined,
    photoCount: p.photoCount,
    participantCount: 0,
    status: "ACTIVE",
    state: p.state as EventState,
    city: p.location.split(",").pop()?.trim() ?? "",
    visibility: p.visibility,
    pricingMode: p.pricingMode,
  };
}

// Live (uploadable) = approved, or approved with a pricing change parked.
export function isOwnedEventLive(p: PhotographerEventSummary): boolean {
  return (
    p.ownedByMe === true &&
    (p.reviewStatus === "approved" || p.reviewStatus === "change_pending")
  );
}

// "Paid · ₱150 · QuickPitik mark" / "Free · your logo" — one line for a
// pricing trio, used for the live settings and for a parked request alike.
export function describePricing(trio: {
  pricingMode: "paid" | "free";
  pricePerPhoto: number | string;
  watermarkPolicy: "platform" | "own" | "none";
}): string {
  if (trio.pricingMode === "free") {
    return `Free · ${trio.watermarkPolicy === "none" ? "no watermark" : "your logo"}`;
  }
  return `Paid · ${formatPrice(Number(trio.pricePerPhoto))} · QuickPitik mark`;
}

export function describePendingChange(change: PendingPricingChange): string {
  return describePricing(change);
}

// One plain line per race check reason code (Plan 2 § 3.1). The server sends
// the codes; this is the only place they become words.
const RACE_CHECK_LINES: Record<string, string> = {
  BAD_WORDS: "The name or notes use a word an admin needs to check.",
  NO_RACE_WORD: "The name has no race word, such as run or marathon.",
  NON_RACE_WORD: "The name has a word that suggests it isn't a race.",
  DATE_OUT_OF_WINDOW: "The race date is too far back or too far ahead for an automatic listing.",
  DAILY_CAP: "You've created several races today, so this one goes to an admin.",
  COVER_FLAGGED: "The cover image needs an admin to check it.",
  COVER_UNCHECKED: "The cover image couldn't be checked automatically.",
  POSSIBLE_DUPLICATE: "It may be the same race as one already listed.",
};

export function raceCheckReasonLine(code: string): string {
  return RACE_CHECK_LINES[code] ?? "An admin will check it.";
}

export const RACE_PENDING_JOIN_LINE =
  "Waiting for an admin: you can join now, and uploads open once an admin approves it.";

// Toast copy for the terminal race outcomes (created, pending_review, joined).
export function raceOutcomeMessage(result: RaceCreateResult): string {
  switch (result.outcome) {
    case "created":
      return "Your race is live on /events.";
    case "pending_review":
      return [
        "Sent for review. Uploads open once an admin approves it.",
        ...result.checkReasons.map(raceCheckReasonLine),
      ].join(" ");
    default:
      return result.event?.reviewStatus === "pending"
        ? RACE_PENDING_JOIN_LINE
        : `You joined ${result.event?.name ?? "the race"}.`;
  }
}

// Short review chip for an owned event card, or for a race the photographer
// submitted (races keep ownedByMe false).
export function ownedEventNote(p: PhotographerEventSummary): string | undefined {
  if (p.eventType && p.submittedByMe) {
    if (p.hidden) return "Hidden from /events";
    switch (p.reviewStatus) {
      case "pending":
        return "Pending review";
      case "rejected":
        return "Sent back";
      case "change_pending":
        return "Change pending";
      default:
        return "Your race";
    }
  }
  if (!p.ownedByMe) return undefined;
  switch (p.reviewStatus) {
    case "pending":
      return "Pending review";
    case "rejected":
      return "Sent back";
    case "change_pending":
      return "Change pending";
    default:
      return `${p.visibility === "unlisted" ? "Unlisted" : "Public"} · ${
        p.pricingMode === "free" ? "Free" : "Paid"
      }`;
  }
}
