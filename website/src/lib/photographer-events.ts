import type { EventState, ListEvent } from "@/app/events/events-browser";
import type {
  PendingPricingChange,
  PhotographerEventSummary,
} from "@/lib/photographer-mock";

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
  return `Paid · ₱${Number(trio.pricePerPhoto)} · QuickPitik mark`;
}

export function describePendingChange(change: PendingPricingChange): string {
  return describePricing(change);
}

// Short review chip for an owned event card.
export function ownedEventNote(p: PhotographerEventSummary): string | undefined {
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
