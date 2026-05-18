"use client";

import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { DashboardActionGrid } from "@/components/dashboard/dashboard-action-grid";
import { SetupJourney } from "@/components/dashboard/setup-journey";
import { useCanUpload } from "@/hooks/use-can-upload";
import {
  usePhotographerEvents,
  usePhotographerPayouts,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import type {
  EventState,
  PhotographerEventSummary,
  PhotographerPayout,
} from "@/lib/photographer-mock";

// Two modes:
//  - Setup mode (first-timer): linear 3-step journey while the photographer
//    is unverified or hasn't uploaded yet. Subsumes the verification banner.
//  - Data mode: action-grid hero + billing + next-up. Earnings preview lives
//    inside the action grid (sparkline card), so a dedicated Earnings slab
//    would duplicate it — billing answers the photographer's other most-
//    asked question ("when am I getting paid?").
export default function DashboardOverviewPage() {
  const gate = useCanUpload();
  const events = usePhotographerEvents();
  // Setup mode is true until the photographer is verified AND has at least
  // one photo uploaded across all their events. Photo count comes from the
  // event_photographer.photo_count column (BE upserts it on every upload).
  const hasAnyUploads = (events ?? []).some((e) => e.photoCount > 0);
  const isSetupMode = gate.kind !== "ok" || !hasAnyUploads;

  if (isSetupMode) {
    return <SetupJourney />;
  }

  return (
    <>
      <DashboardActionGrid />
      <BillingGlance />
      <NextUpGlance />
    </>
  );
}

function BillingGlance() {
  const payouts = usePhotographerPayouts() ?? [];
  const next = pickNextScheduled(payouts);
  const inReviewTotal = payouts
    .filter((p) => p.status === "pending")
    .reduce((sum, p) => sum + p.amount, 0);

  return (
    <Slab
      id="billing"
      number="01"
      title="Billing"
      trailing={
        <Link
          href={ROUTES.DASHBOARD_BILLING}
          className="hover:text-ink transition-colors inline-flex items-center gap-1.5 group"
        >
          Open billing
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </Link>
      }
    >
      {next ? (
        <>
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
            Next payout
          </p>
          <p className="font-display text-5xl md:text-6xl font-semibold tracking-tight text-fresh tnum mt-3 leading-none">
            ₱{next.amount.toLocaleString()}
          </p>
          <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mt-4 tnum">
            {formatLongDate(next.settledAt)} ·{" "}
            {payoutMethodLabel(next.method)}
          </p>

          {inReviewTotal > 0 && (
            <p className="font-sans text-sm text-ink-soft mt-6 max-w-md">
              ₱{inReviewTotal.toLocaleString()} still in review from last
              cycle.
            </p>
          )}
        </>
      ) : (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
            No payouts scheduled.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            Sales accrue weekly — your first payout cycle will show up here
            after your first sale settles.
          </p>
        </div>
      )}
    </Slab>
  );
}

function pickNextScheduled(
  payouts: ReadonlyArray<PhotographerPayout>,
): PhotographerPayout | undefined {
  return [...payouts]
    .filter((p) => p.status === "scheduled")
    .sort((a, b) => a.settledAt.localeCompare(b.settledAt))[0];
}

function payoutMethodLabel(method: PhotographerPayout["method"]): string {
  if (method === "gcash") return "GCash";
  if (method === "maya") return "Maya";
  if (method === "gotyme") return "GoTyme";
  return method;
}

function NextUpGlance() {
  // Prefer a currently-live event over the next upcoming one — if the
  // photographer has a coverage today, that's what they came to the
  // dashboard for. Both route into /upload/[eventId]: live opens the
  // dropzone, upcoming lands on the "Uploads open on race day" panel.
  // /dashboard/events/[id] 404s for events without uploads, so we
  // deliberately avoid that path here.
  const events = usePhotographerEvents() ?? [];
  const live = events.find((e) => e.state === "live");
  const upcoming = [...events]
    .filter((e) => e.state === "upcoming")
    .sort((a, b) => a.date.localeCompare(b.date))[0];
  const featured = live ?? upcoming;
  const isLive = !!live;

  return (
    <Slab
      id="next-up"
      number="02"
      title={isLive ? "Live now" : "Next up"}
      trailing={
        <Link
          href={ROUTES.DASHBOARD_EVENTS}
          className="hover:text-ink transition-colors inline-flex items-center gap-1.5 group"
        >
          All events
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </Link>
      }
    >
      {!featured ? (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
            Nothing on the calendar.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            Schedule your next coverage to start showing up here.
          </p>
        </div>
      ) : (
        <ul className="border-y border-line divide-y divide-line">
          <li>
            <FeaturedEventRow event={featured} />
          </li>
        </ul>
      )}
    </Slab>
  );
}

function FeaturedEventRow({ event }: { event: PhotographerEventSummary }) {
  const isLive = event.state === "live";
  const cta = isLive ? "Upload" : "Open";

  return (
    <Link
      href={`${ROUTES.UPLOAD}/${event.id}`}
      className="group block py-5 md:py-6 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum flex items-center gap-2 flex-wrap">
            <span>{formatLongDate(event.date, true)}</span>
            <span className="text-slate-soft">·</span>
            <StateChip state={event.state} />
          </p>
          <h3 className="font-display text-lg md:text-xl font-medium tracking-tight text-ink mt-2 truncate group-hover:text-ink-soft transition-colors">
            {event.name}
          </h3>
        </div>
        <span className="font-sans text-sm text-ink group-hover:text-fresh transition-colors inline-flex items-center gap-1.5">
          {cta}
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </span>
      </div>
    </Link>
  );
}

const STATE_LABEL: Record<EventState, string> = {
  live: "LIVE",
  open: "OPEN",
  upcoming: "UPCOMING",
  past: "ARCHIVED",
};

function StateChip({ state }: { state: EventState }) {
  if (state === "live") {
    return (
      <span className="inline-flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh breathe"
        />
        <span className="text-ink">LIVE</span>
      </span>
    );
  }
  return <span>{STATE_LABEL[state]}</span>;
}
