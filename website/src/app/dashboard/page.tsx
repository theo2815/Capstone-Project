"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { DashboardActionGrid } from "@/components/dashboard/dashboard-action-grid";
import { SetupJourney } from "@/components/dashboard/setup-journey";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton, TileSkeleton } from "@/components/ui/skeleton";
import { useCanUpload } from "@/hooks/use-can-upload";
import {
  usePhotographerEvents,
  usePhotographerPayoutBalance,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import type {
  EventState,
  PhotographerEventSummary,
} from "@/lib/photographer-mock";
import { cn } from "@/lib/utils";

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
  // PF-8 (2026-05-27): treat `events === null` as "still loading" so verified
  // photographers don't flash the 3-step SetupJourney before data arrives.
  const eventsLoaded = events !== null;
  const hasAnyUploads = (events ?? []).some((e) => e.photoCount > 0);
  const isSetupMode = eventsLoaded && (gate.kind !== "ok" || !hasAnyUploads);

  // Neither branch is safe to render before `events` resolves — gating only
  // one way (PF-8) traded the verified photographer's SetupJourney flash for
  // a first-timer's action-grid flash. Hold a skeleton until the fork is
  // decidable so nobody sees the wrong dashboard.
  if (!eventsLoaded) {
    return <OverviewSkeleton />;
  }

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

// Deliberately neutral: the fork above can resolve to either SetupJourney or
// the action grid, and both open with a kicker + headline. Anything more
// specific would make the wrong destination flash in skeleton form instead of
// in real content — the same bug one layer down.
function OverviewSkeleton() {
  return (
    <section className="pb-12 md:pb-16" aria-busy="true">
      <Skeleton className="h-3 w-32" />
      <Skeleton className="h-9 md:h-11 w-72 max-w-full mt-4" />
      <div className="mt-8 md:mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
        {[0, 1, 2, 3].map((i) => (
          <TileSkeleton key={i} aspectRatio="h-44 md:h-48" />
        ))}
      </div>
    </section>
  );
}

// Condensed mirror of the /dashboard/billing Request-payout hero. Three
// states — open request / above threshold / below threshold. The full UI
// (request modal, withdraw, recent cycles) lives on /dashboard/billing;
// this is a glance card with an "Open billing" CTA.
function BillingGlance() {
  const balance = usePhotographerPayoutBalance();

  return (
    <Slab
      id="billing"
      number="01"
      title="Billing"
      trailing={
        <Link
          href={ROUTES.DASHBOARD_BILLING}
          className="text-ink hover:text-fresh transition-colors inline-flex items-center gap-1.5 group"
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
      {balance === null ? (
        <div>
          <Skeleton className="h-3 w-28" />
          <Skeleton className="h-12 md:h-14 w-64 mt-3" />
          <Skeleton className="h-3 w-44 mt-4" />
        </div>
      ) : balance.hasOpenRequest && balance.openRequest ? (
        <OpenRequestGlance request={balance.openRequest} />
      ) : (
        <AvailableGlance
          unpaidBalance={balance.unpaidBalance}
          minimum={balance.minimum}
        />
      )}
    </Slab>
  );
}

function OpenRequestGlance({
  request,
}: {
  request: NonNullable<
    ReturnType<typeof usePhotographerPayoutBalance>
  >["openRequest"] & object;
}) {
  const stage: "pending_review" | "approved" | "held" =
    request.status === "held"
      ? "held"
      : request.settledAt
        ? "approved"
        : "pending_review";
  const stageLabel = {
    pending_review: "Pending review",
    approved: "Approved · payment incoming",
    held: "Held — needs attention",
  } as const;
  const stageTone = {
    pending_review: "text-ink",
    approved: "text-fresh",
    held: "text-error",
  } as const;

  return (
    <>
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft flex items-center gap-2 flex-wrap">
        <span>Payout request</span>
        <span className="text-slate-soft">·</span>
        <span className={stageTone[stage]}>{stageLabel[stage]}</span>
      </p>
      <p className="font-display text-5xl md:text-6xl font-extrabold tracking-tight text-ink tnum mt-3 leading-none">
        ₱{request.amount.toLocaleString()}
      </p>
      {stage === "approved" && request.settledAt && (
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mt-4 tnum">
          Approved {formatLongDate(request.settledAt)}
        </p>
      )}
      {stage === "held" && request.holdReason && (
        <p className="font-sans text-sm text-ink-soft mt-4 max-w-md">
          <span className="text-error">Reason: </span>
          {request.holdReason}
        </p>
      )}
    </>
  );
}

function AvailableGlance({
  unpaidBalance,
  minimum,
}: {
  unpaidBalance: number;
  minimum: number;
}) {
  const eligible = unpaidBalance >= minimum;

  if (!eligible && unpaidBalance === 0) {
    return (
      <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
        <p className="font-display text-2xl md:text-3xl font-extrabold tracking-tight text-ink">
          No earnings yet.
        </p>
        <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
          Your first sale opens up the request flow — reach ₱
          {minimum.toLocaleString()} to request a payout.
        </p>
      </div>
    );
  }

  return (
    <>
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        Available to request
      </p>
      <p
        className={cn(
          "font-display text-5xl md:text-6xl font-extrabold tracking-tight tnum mt-3 leading-none",
          eligible ? "text-fresh" : "text-slate",
        )}
      >
        ₱{unpaidBalance.toLocaleString()}
      </p>
      <p className="font-sans text-sm text-ink-soft mt-4 max-w-md">
        {eligible ? (
          <>
            Ready to request — open billing to confirm and submit.
          </>
        ) : (
          <>
            Reach{" "}
            <span className="font-mono tnum">₱{minimum.toLocaleString()}</span>{" "}
            in unpaid earnings to request a payout.
          </>
        )}
      </p>
    </>
  );
}

function NextUpGlance() {
  // Lists every upcoming event the photographer is assigned to, soonest
  // first. Rows route into /upload/[eventId] which shows the "Uploads
  // open on race day" panel until the event flips to live state.
  // Live events are intentionally NOT surfaced here — the dashboard's
  // action grid handles "happening now"; this section is forward-looking.
  const events = usePhotographerEvents();
  const isLoading = events === null;
  const upcoming = useMemo(
    () =>
      (events ?? [])
        .filter((e) => e.state === "upcoming")
        .sort((a, b) => a.date.localeCompare(b.date)),
    [events],
  );
  const [shown, setShown] = useState(PAGE_SIZE.UPCOMING_INITIAL);

  return (
    <Slab
      id="next-up"
      number="02"
      title="Next up"
      trailing={
        <Link
          href={ROUTES.DASHBOARD_EVENTS}
          className="text-ink hover:text-fresh transition-colors inline-flex items-center gap-1.5 group"
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
      {isLoading ? (
        <ul className="border-y border-line divide-y divide-line">
          {[0, 1].map((i) => (
            <li key={i} className="py-5 md:py-6">
              <Skeleton className="h-3 w-40" />
              <Skeleton className="h-5 md:h-6 w-64 mt-2" />
            </li>
          ))}
        </ul>
      ) : upcoming.length === 0 ? (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-extrabold tracking-tight text-ink">
            No upcoming coverage.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            Once organizers assign you to a future event, it shows up here.
          </p>
        </div>
      ) : (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {upcoming.slice(0, shown).map((event) => (
              <li key={event.id}>
                <FeaturedEventRow event={event} />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={Math.min(shown, upcoming.length)}
            total={upcoming.length}
            increment={PAGE_SIZE.UPCOMING_INCREMENT}
            onLoadMore={() =>
              setShown((n) => n + PAGE_SIZE.UPCOMING_INCREMENT)
            }
          />
        </>
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
      className="group block py-5 md:py-6 px-3 -mx-3 rounded-lg hover:bg-bone-deep/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum flex items-center gap-2 flex-wrap">
            <span>{formatLongDate(event.date, true)}</span>
            <span className="text-slate-soft">·</span>
            <StateChip state={event.state} />
          </p>
          <h3 className="font-display text-lg md:text-xl font-bold tracking-tight text-ink mt-2 truncate group-hover:text-fresh transition-colors">
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
