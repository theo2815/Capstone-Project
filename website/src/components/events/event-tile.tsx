"use client";

import Link from "next/link";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { SaveButton } from "@/components/events/save-button";
import { ROUTES } from "@/lib/constants";
import { canUploadToEvent } from "@/lib/event-catalog";

// Reused by /events (browse), /dashboard/upload (upload), and
// /dashboard/events (manage). Discriminated union mirrors <PhotoPreviewCard>.
// Browse keeps the runner-facing SaveButton + photo-count footer; upload
// drops the SaveButton, routes to the upload form gated on verification, and
// trades photo count for participant count; manage shows photographer-side
// stats (photos uploaded · sold) and routes to the focused share page.

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

type EventTileUploadProps = {
  mode: "upload";
  event: ListEvent;
  index?: number;
  canUpload: boolean;
};

type EventTileManageProps = {
  mode: "manage";
  event: ListEvent;
  index?: number;
  photoCount: number;
  salesCount: number;
};

export type EventTileProps =
  | EventTileBrowseProps
  | EventTileUploadProps
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
            className="absolute inset-0 w-full h-full object-cover"
          />
        ) : (
          <>
            <div className="absolute inset-0 flex items-center justify-center px-6">
              <span className="font-display text-bone/25 text-2xl md:text-3xl font-medium tracking-tight text-center leading-tight">
                {event.name}
              </span>
            </div>
            <span className="absolute bottom-3 right-3 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/35">
              Banner · soon
            </span>
          </>
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
        {mode === "browse" && <BrowseFooter isUpcoming={isUpcoming} />}
        {mode === "upload" && (
          <UploadFooter
            event={event}
            canUpload={(props as EventTileUploadProps).canUpload}
          />
        )}
        {mode === "manage" && <ManageFooter />}
      </div>
    </>
  );

  if (isBrowse) {
    if (isUpcoming) {
      return (
        <div
          aria-label={`${event.name} — opens on race day`}
          className="group block rounded-2xl border border-line bg-surface shadow-[var(--shadow-card)] overflow-hidden"
          style={animationStyle}
        >
          {body}
        </div>
      );
    }
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

  if (mode === "upload") {
    // every state is clickable. If verification is incomplete, route to
    // settings instead and reflect that in the CTA + aria label.
    const { canUpload } = props as EventTileUploadProps;
    const target = canUpload
      ? `${ROUTES.UPLOAD}/${event.id}`
      : ROUTES.DASHBOARD_SETTINGS;
    const aria = canUpload
      ? `Upload photos to ${event.name}`
      : `Finish settings to upload to ${event.name}`;

    return (
      <Link
        href={target}
        aria-label={aria}
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
  // is just the action affordance. Upcoming events are non-interactive — show
  // the wait copy instead of an arrow so runners aren't tempted to click.
  return (
    <div className="mt-6 pt-4 border-t border-line flex items-center justify-end">
      {isUpcoming ? (
        <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Opens on race day
        </span>
      ) : (
        <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink group-hover:text-fresh transition-colors">
          Open →
        </span>
      )}
    </div>
  );
}

function UploadFooter({
  event,
  canUpload,
}: {
  event: ListEvent;
  canUpload: boolean;
}) {
  // Verification-incomplete is the most actionable issue, so it wins over
  // the date-based hints. Once verified, the date drives copy:
  //   upcoming      → opens on race day
  //   live (grace)  → upload →
  //   open / past   → window closed (4-day grace expired)
  const inGrace = canUploadToEvent(event.date);
  let cta: string;
  let muted = false;
  if (!canUpload) {
    cta = "Finish settings →";
  } else if (event.state === "upcoming") {
    cta = "Opens on race day →";
  } else if (inGrace) {
    cta = "Upload →";
  } else {
    cta = "Upload window closed";
    muted = true;
  }

  return (
    <div className="mt-6 pt-4 border-t border-line flex items-center justify-end">
      <span
        className={
          muted
            ? "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate"
            : "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink group-hover:text-fresh transition-colors"
        }
      >
        {cta}
      </span>
    </div>
  );
}

function ManageFooter() {
  return (
    <div className="mt-6 pt-4 border-t border-line flex items-center justify-end">
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
