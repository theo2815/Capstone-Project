"use client";

import Link from "next/link";
import type { EventState, ListEvent } from "@/types/event";
import { SaveButton } from "@/components/events/save-button";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import { ROUTES } from "@/lib/constants";

// Reused by /events (browse) and the photographer profiles (browse);
// manage mode is unused since /dashboard/events moved to rows. Discriminated union mirrors <PhotoPreviewCard>.
// Browse keeps the runner-facing SaveButton + photo-count footer; manage shows
// photographer-side stats (photos uploaded · sold) and routes to the focused
// share page. The upload picker uses DashboardEventRow + uploadCta.

type EventTileBrowseProps = {
  mode?: "browse";
  event: ListEvent;
  index?: number;
  /** Override the default `/events/{slug}` link target. Used by the public
   *  photographer profile to route into `/{handle}/events/{slug}` instead. */
  hrefOverride?: string;
  /** When true, render the SaveButton in its non-interactive "runners only"
   *  state — same visual as default unsaved, but click is a no-op and the
   *  tooltip swaps. Used on /{handle} owner-preview where saving your own
   *  event isn't a meaningful action. Public visitors leave this off so
   *  runners can still save. */
  saveDisabled?: boolean;
};

type EventTileManageProps = {
  mode: "manage";
  event: ListEvent;
  index?: number;
  photoCount: number;
  salesCount: number;
  /** Review / visibility chip for an owned event ("Pending review",
   *  "Unlisted · Free"). Admin events carry none. */
  note?: string;
};

export type EventTileProps =
  | EventTileBrowseProps
  | EventTileManageProps;

export function EventTile(props: EventTileProps) {
  const { event } = props;
  const index = props.index ?? 0;
  const mode = props.mode ?? "browse";
  const isBrowse = mode === "browse";
  const isUpcoming = event.state === "upcoming";

  const dateLabel = formatShortDate(event.date);
  const cityUpper = event.city.toUpperCase();

  const animationStyle = {
    animation: `fade-up 0.5s ${0.05 * index + 0.05}s both`,
    opacity: 0,
  };

  const body = (
    <>
      <div className="relative aspect-[4/3] bg-ink overflow-hidden">
        {event.bannerUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={event.bannerUrl}
            alt=""
            aria-hidden="true"
            loading="lazy"
            decoding="async"
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          // No cover yet: the QuickPitik logo instead of a faint name and "Banner · soon".
          <LogoPlaceholder tone="dark" size="lg" className="absolute inset-0" />
        )}
        <StatusChip state={event.state} />
        {isBrowse && (
          <SaveButton
            eventId={event.id}
            event={{
              id: event.id,
              slug: event.slug,
              name: event.name,
              date: event.date,
              state: event.state,
              bannerUrl: event.bannerUrl ?? null,
              location: event.location,
            }}
            variant="card"
            disabled={(props as EventTileBrowseProps).saveDisabled}
          />
        )}
      </div>
      <div className="p-6 md:p-7">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mb-3">
          <span className="tnum">{dateLabel}</span> · {cityUpper}
        </p>
        <h3 className="font-display text-2xl md:text-3xl font-extrabold tracking-tight leading-tight text-ink">
          {event.name}
        </h3>
        <p className="mt-3 font-sans text-sm md:text-base text-ink-soft">
          {event.location}
        </p>
        {/* Some photographer in this race charges above the race price (AA-R14). */}
        {event.priceVaries && event.pricePerPhoto != null && (
          <span className="mt-3 inline-flex items-center rounded-full border border-line bg-bone-deep px-2.5 py-1 font-mono uppercase tracking-[0.14em] text-[13px] text-slate whitespace-nowrap tnum">
            From ₱{event.pricePerPhoto}
          </span>
        )}
        {mode === "browse" && <BrowseFooter isUpcoming={isUpcoming} />}
        {props.mode === "manage" && <ManageFooter note={props.note} />}
      </div>
    </>
  );

  if (isBrowse) {
    const hrefOverride = (props as EventTileBrowseProps).hrefOverride;
    return (
      <Link
        href={hrefOverride ?? `/events/${event.slug}`}
        aria-label={`Open ${event.name}`}
        className="group block rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)] overflow-hidden transition-all duration-300 hover:border-fresh/50 hover:-translate-y-1 hover:shadow-[var(--shadow-lift)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-paper"
        style={animationStyle}
      >
        {body}
      </Link>
    );
  }

  // manage mode — links to the focused share page at /dashboard/events/[id].
  return (
    <Link
      href={`${ROUTES.DASHBOARD_EVENTS}/${event.id}`}
      aria-label={`Open ${event.name}`}
      className="group block rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)] overflow-hidden transition-all duration-300 hover:border-fresh/50 hover:-translate-y-1 hover:shadow-[var(--shadow-lift)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-paper"
      style={animationStyle}
    >
      {body}
    </Link>
  );
}

function BrowseFooter({ isUpcoming }: { isUpcoming: boolean }) {
  // Card body already carries date · city + title + location, so the footer
  // is just the action affordance. Upcoming events are clickable too — the
  // detail page lets runners turn on the photos-ready alert before race day.
  return (
    <div className="mt-6 pt-4 border-t border-line flex items-center justify-end">
      <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink group-hover:text-fresh transition-colors">
        {isUpcoming ? "Get notified →" : "Open →"}
      </span>
    </div>
  );
}

function ManageFooter({ note }: { note?: string }) {
  return (
    <div className="mt-6 pt-4 border-t border-line flex items-center justify-between gap-4">
      <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate truncate">
        {note}
      </span>
      <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink group-hover:text-fresh transition-colors">
        Open →
      </span>
    </div>
  );
}

export function StatusChip({ state }: { state: EventState }) {
  if (state === "live") {
    return (
      <div className="absolute top-4 left-5 flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh breathe"
        />
        <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh">
          Photos uploading
        </span>
      </div>
    );
  }
  if (state === "upcoming") {
    return (
      <div className="absolute top-4 left-5 flex items-center gap-2.5">
        <span aria-hidden="true" className="size-1.5 rounded-full bg-fresh" />
        <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/85">
          Save the date
        </span>
      </div>
    );
  }
  if (state === "open") {
    return (
      <div className="absolute top-4 left-5 flex items-center gap-2.5">
        <span aria-hidden="true" className="size-1.5 rounded-full bg-bone/85" />
        <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/85">
          Photos ready
        </span>
      </div>
    );
  }
  return (
    <div className="absolute top-4 left-5 flex items-center gap-2.5">
      <span aria-hidden="true" className="size-1.5 rounded-full bg-bone/40" />
      <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/55">
        Archive
      </span>
    </div>
  );
}

function formatShortDate(iso: string) {
  const d = new Date(iso + "T00:00:00");
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}
