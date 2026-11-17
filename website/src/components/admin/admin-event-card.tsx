"use client";

import Link from "next/link";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import type { EventState, ListEvent } from "@/types/event";
import { ROUTES } from "@/lib/constants";
import type { EventReviewStatus } from "@/lib/photographer-mock";

// Admin-side variant of <EventTile>. Reuses the dark banner + StatusChip
// from the runner-facing tile but trades navigation for a row of explicit
// admin affordances: Edit · Delete · View ↗.
//
// Lifecycle state is now derived from `event.date` (see
// `lib/event-catalog.ts`), so the previous "Move to" dropdown is gone.
// Card body shows only Title + Date + City + Location — same fields the
// runner-facing tile shows.

// Photographer-owned event (V46): who owns it and where it sits in review.
// Admin-created events pass nothing and render exactly as before.
export interface AdminEventReview {
  owner: string;
  status: EventReviewStatus;
  visibility: "public" | "unlisted";
  // V89 admin "Show on /events" switch.
  listed: boolean;
}

const REVIEW_LABEL: Record<EventReviewStatus, string> = {
  pending: "Pending review",
  rejected: "Sent back",
  change_pending: "Pricing change pending",
  approved: "Live",
};

interface AdminEventCardProps {
  event: ListEvent;
  index?: number;
  review?: AdminEventReview;
  pendingLabel?: string | null;
  onEdit: (event: ListEvent) => void;
  onDelete: (event: ListEvent) => void;
  onSetListing?: (event: ListEvent, listed: boolean) => void;
  /** AA-R17b shared race: whether it is hidden from /events. Undefined = not a race. */
  raceHidden?: boolean;
  onRaceListing?: (event: ListEvent, listed: boolean) => void;
}

export function AdminEventCard({
  event,
  review,
  pendingLabel,
  onEdit,
  onDelete,
  onSetListing,
  raceHidden,
  onRaceListing,
}: AdminEventCardProps) {
  const dateLabel = formatShortDate(event.date);
  // A pending / sent-back submission is a DRAFT: no public page to view,
  // so the link points at the queue instead.
  const awaitingReview = review !== undefined && event.status === "DRAFT";
  // Only a live Public photographer event can go on /events; admin events always are.
  const canList = review !== undefined && !awaitingReview && review.visibility === "public" && onSetListing;

  return (
    // Proposal 12: one compact list row instead of a 4:3 cover card, so a
    // screen holds 8–9 races. Same fields and actions as before.
    <article
      aria-busy={!!pendingLabel}
      className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 transition-colors hover:bg-bone-deep/50"
    >
      <div className="relative w-[88px] h-[56px] shrink-0 rounded-lg overflow-hidden bg-bone-deep flex items-center justify-center">
        {event.bannerUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={event.bannerUrl}
            alt=""
            aria-hidden="true"
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <LogoPlaceholder size="sm" className="absolute inset-0" />
        )}
      </div>
      <div className="min-w-0 flex-1 basis-[240px]">
        <h3 className="font-display text-base font-semibold text-ink">
          {event.name}
        </h3>
        <p className="font-sans text-sm text-slate">
          <span className="tnum">{dateLabel}</span> · {event.city}
          {event.location && event.location !== event.city && ` · ${event.location}`}
        </p>
        {review && (
          <p className="font-sans text-sm text-slate">
            Photographer event · {review.owner} · {REVIEW_LABEL[review.status]}
          </p>
        )}
        {raceHidden && (
          // Review F36: say what hiding does and does not do.
          <p className="font-sans text-sm text-ink-soft">
            <span className="font-semibold text-ink">Hidden</span>. Off /events and the upload pickers. Its link still opens, and it still shows on its photographers&apos; public profiles.
          </p>
        )}
      </div>
      <span className={`inline-flex items-center gap-1.5 font-sans text-sm w-[120px] shrink-0 ${STATE_TEXT[event.state]}`}>
        <span aria-hidden="true" className="size-[7px] rounded-full bg-current" />
        {STATE_LABEL[event.state]}
      </span>
      {/* Fixed width from md so the status column lines up row to row. */}
      <div className="flex items-center justify-end gap-1 flex-wrap ml-auto md:w-[23rem]">
        <button
          type="button"
          onClick={() => onEdit(event)}
          disabled={!!pendingLabel}
          className={actionBtn}
        >
          Edit
        </button>
        <button
          type="button"
          onClick={() => onDelete(event)}
          disabled={!!pendingLabel}
          className={dangerBtn}
          aria-label={`Delete ${event.name}`}
        >
          Delete
        </button>
        {canList && (
          <label className="inline-flex items-center gap-2 min-h-[44px] px-3 font-sans text-sm text-ink cursor-pointer has-[:disabled]:opacity-50 has-[:disabled]:cursor-not-allowed">
            <input
              type="checkbox"
              className="size-5 accent-fresh"
              checked={review.listed}
              disabled={!!pendingLabel}
              onChange={(e) => onSetListing(event, e.target.checked)}
            />
            Show on /events
          </label>
        )}
        {raceHidden !== undefined && onRaceListing && (
          <button
            type="button"
            onClick={() => onRaceListing(event, raceHidden)}
            disabled={!!pendingLabel}
            className={actionBtn}
          >
            {raceHidden ? "Relist" : "Hide"}
          </button>
        )}
        {pendingLabel && <p role="status" className="font-sans text-sm text-slate px-2">{pendingLabel}</p>}
        {awaitingReview ? (
          <Link href={ROUTES.ADMIN_EVENT_REQUESTS} className={linkBtn}>
            Review →
          </Link>
        ) : (
          <Link href={`/events/${event.slug}`} target="_blank" rel="noreferrer" className={linkBtn}>
            View ↗
          </Link>
        )}
      </div>
    </article>
  );
}

const STATE_LABEL: Record<EventState, string> = {
  live: "Live now",
  upcoming: "Upcoming",
  open: "Open for sale",
  past: "Archived",
};
// The dot takes the text colour; the word carries the meaning, not the colour.
const STATE_TEXT: Record<EventState, string> = {
  live: "text-fresh",
  upcoming: "text-slate",
  open: "text-ink",
  past: "text-slate-soft",
};


// 44px tall so the row's actions are comfortable to tap (they were 20px).
// Sentence case (proposal 11): mono uppercase is kept for labels only.
const actionBtn =
  "inline-flex items-center min-h-[44px] px-3 font-sans text-sm font-semibold text-ink hover:text-fresh transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
const linkBtn =
  "inline-flex items-center min-h-[44px] px-3 font-sans text-sm text-slate hover:text-ink transition-colors";

// Delete is destructive, so it reads in the error colour (house BTN_DANGER tone).
const dangerBtn = actionBtn.replace("text-ink hover:text-fresh", "text-error hover:text-error/80");

function formatShortDate(iso: string) {
  const d = new Date(iso + "T00:00:00");
  const month = d.toLocaleString("en-US", { month: "short" });
  return `${month} ${d.getDate()}, ${d.getFullYear()}`;
}
