"use client";

import { useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { Sparkline } from "@/components/dashboard/sparkline";
import { PlatformCutModal } from "@/components/dashboard/platform-cut-modal";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  usePhotographerEarnings,
  usePhotographerPerEventEarnings,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { PAGE_SIZE } from "@/lib/pagination-config";
import type { PerEventEarning } from "@/lib/api-photographer-earnings";
import { formatPrice } from "@/lib/utils";

export default function EarningsPage() {
  return (
    <>
      <LifetimeSlab />
      <BreakdownSlab />
      <PerEventSlab />
    </>
  );
}

function LifetimeSlab() {
  // GET /me/photographer/earnings — null while in flight. No mock fallback:
  // the previous `?? PHOTOGRAPHER_EARNINGS` made a BE 404 / slow start render
  // seeded fake numbers as user data. Skeleton on null is the honest signal.
  const e = usePhotographerEarnings();
  const [cutModalOpen, setCutModalOpen] = useState(false);

  if (!e) {
    return (
      <Slab
        id="lifetime"
        number="01"
        title="Lifetime"
        caption="After platform cut"
      >
        <div>
          <Skeleton className="h-3 w-32" />
          <Skeleton className="h-12 md:h-16 w-72 md:w-96 mt-3" />
          <Skeleton className="h-4 w-48 mt-4" />
        </div>
        <div className="mt-10">
          <Skeleton className="h-3 w-28" />
          <Skeleton className="h-3 w-56 mt-2" />
          <Skeleton className="h-16 md:h-20 w-full mt-4" />
        </div>
      </Slab>
    );
  }

  return (
    <Slab
      id="lifetime"
      number="01"
      title="Lifetime"
      caption="After platform cut"
    >
      <div>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          Lifetime kept
        </p>
        <p className="font-display text-5xl md:text-7xl font-semibold tracking-tight text-fresh tnum mt-3 leading-none">
          ₱{e.lifetimeKept.toLocaleString()}
        </p>
        <button
          type="button"
          onClick={() => setCutModalOpen(true)}
          className="mt-4 inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone rounded-sm"
        >
          <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
            How is this calculated?
          </span>
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </button>
      </div>

      <div className="mt-10">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          12-week trend
        </p>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1.5">
          Hover or tap a bar for week details
        </p>
        <div className="mt-4">
          <Sparkline
            data={e.weeklySeries}
            ariaLabel="Weekly earnings, last 12 weeks"
            interactive
          />
        </div>
      </div>

      <PlatformCutModal
        isOpen={cutModalOpen}
        onClose={() => setCutModalOpen(false)}
      />
    </Slab>
  );
}

function BreakdownSlab() {
  const e = usePhotographerEarnings();
  if (!e) {
    return (
      <Slab id="breakdown" number="02" title="Breakdown" caption="Current period">
        <div className="grid grid-cols-3 gap-4 md:gap-8">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="border-l border-line pl-4 md:pl-6 first:border-0 first:pl-0"
            >
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-8 md:h-10 w-24 mt-3" />
              <Skeleton className="h-4 w-16 mt-3" />
            </div>
          ))}
        </div>
      </Slab>
    );
  }
  return (
    <Slab id="breakdown" number="02" title="Breakdown" caption="Current period">
      <div className="grid grid-cols-3 gap-4 md:gap-8">
        <BreakdownStat
          kicker="This week"
          value={`₱${e.thisWeek.toLocaleString()}`}
          caption={`+${e.thisWeekSold} sold`}
        />
        <BreakdownStat
          kicker="This month"
          value={`₱${e.thisMonth.toLocaleString()}`}
          caption={`+${e.thisMonthSold} sold`}
        />
        <BreakdownStat
          kicker="Payout pending"
          value={`₱${e.payoutPending.toLocaleString()}`}
          caption={`Releases ${formatLongDate(e.payoutScheduledFor, true)}`}
        />
      </div>

      <div className="mt-10">
        <Link
          href={ROUTES.DASHBOARD_BILLING}
          className="inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors group"
        >
          <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
            View payouts and transactions
          </span>
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </Link>
      </div>
    </Slab>
  );
}

function BreakdownStat({
  kicker,
  value,
  caption,
}: {
  kicker: string;
  value: string;
  caption: string;
}) {
  return (
    <div className="border-l border-line pl-4 md:pl-6 first:border-0 first:pl-0">
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
        {kicker}
      </p>
      <p className="font-display font-medium tracking-tight tnum text-2xl md:text-4xl text-ink mt-3 leading-none">
        {value}
      </p>
      <p className="font-sans text-sm text-slate mt-3">{caption}</p>
    </div>
  );
}

function PerEventSlab() {
  const livePerEvent = usePhotographerPerEventEarnings();
  const events = livePerEvent.items;

  if (livePerEvent.isLoading) {
    return (
      <Slab id="per-event" number="03" title="Per-event">
        <ul className="border-y border-line divide-y divide-line">
          {[0, 1, 2].map((i) => (
            <li key={i} className="py-5 md:py-6">
              <Skeleton className="h-3 w-28" />
              <Skeleton className="h-5 md:h-6 w-64 mt-2" />
              <Skeleton className="h-4 w-44 mt-2" />
            </li>
          ))}
        </ul>
      </Slab>
    );
  }

  return (
    <Slab
      id="per-event"
      number="03"
      title="Per-event"
    >
      {livePerEvent.error ? (
        <div role="alert" className="font-sans text-base text-slate">
          <p>Couldn&apos;t load event earnings.</p>
          <button type="button" onClick={livePerEvent.refetch} className="mt-4 min-h-[44px] underline">Retry</button>
        </div>
      ) : events.length === 0 ? (
        <p className="font-sans text-base text-slate max-w-md">
          No event earnings yet.
        </p>
      ) : (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {events.map((event) => (
              <li key={event.eventId}>
                <PerEventRow event={event} />
              </li>
            ))}
          </ul>
          {livePerEvent.appendError && <p role="alert" className="mt-4 font-sans text-sm text-slate">Couldn&apos;t load more event earnings. Try again.</p>}
          <LoadMoreButton
            shown={events.length}
            total={livePerEvent.hasNextPage ? livePerEvent.total : events.length}
            increment={PAGE_SIZE.EARNINGS_INCREMENT}
            onLoadMore={livePerEvent.fetchNextPage}
            isLoading={livePerEvent.isFetchingNextPage}
            terminalLabel={!livePerEvent.hasNextPage && events.length < livePerEvent.total ? "All available event earnings loaded" : undefined}
          />
        </>
      )}
    </Slab>
  );
}

function PerEventRow({
  event,
}: {
  event: PerEventEarning;
}) {
  return (
    <Link
      href={`${ROUTES.DASHBOARD_EVENTS}/${event.eventId}?tab=purchased`}
      className="group block min-h-[44px] py-5 md:py-6 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
            {formatLongDate(event.eventDate, true)}
          </p>
          <h3 className="font-display text-lg md:text-xl font-medium tracking-tight text-ink mt-2 truncate group-hover:text-ink-soft transition-colors">
            {event.eventName}
          </h3>
          {event.deletedAt && <p className="mt-2 font-mono text-sm uppercase tracking-wide text-slate">Removed</p>}
          <p className="font-sans text-sm text-slate mt-2 tnum">
            <span className="font-mono text-ink-soft">{event.purchaseCount}</span>
            {` purchase${event.purchaseCount === 1 ? "" : "s"} · `}
            <span className="font-mono text-ink-soft">{event.photoCount}</span>
            {" photos"}
          </p>
        </div>
        <div className="flex items-center gap-4 md:flex-col md:items-end md:gap-1 shrink-0">
          <p className="font-mono tnum font-medium text-ink text-lg md:text-xl">
            {formatPrice(event.revenueKept)}
          </p>
        </div>
      </div>
    </Link>
  );
}
