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
import type { PhotographerEventSummary } from "@/lib/photographer-mock";
import { cn } from "@/lib/utils";

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
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
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
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          12-week trend
        </p>
        <p className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate-soft mt-1.5">
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
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
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
  // GET /me/photographer/earnings/per-event response collapses to the
  // PhotographerEventSummary shape so the existing render code works
  // unchanged.
  const livePerEvent = usePhotographerPerEventEarnings();
  const rawEvents = livePerEvent
    ? livePerEvent.map((row) => ({
        id: row.eventId,
        slug: row.eventName,
        name: row.eventName,
        date: row.eventDate,
        location: "",
        state: "open" as const,
        photoCount: row.photoCount,
        salesCount: row.salesCount,
        revenueKept: row.revenueKept,
      }))
    : null;
  const isLoading = livePerEvent === null;
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.EARNINGS_INITIAL);

  if (isLoading || !rawEvents) {
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

  // Sort events by lifetime revenue kept, descending. Skip events with zero
  // (upcoming events not yet covered) so the table shows revenue-bearing
  // entries only. The full list lives at /dashboard/events.
  const events = [...rawEvents]
    .filter((e) => e.revenueKept > 0)
    .sort((a, b) => b.revenueKept - a.revenueKept);

  const totalKept = events.reduce((sum, e) => sum + e.revenueKept, 0);
  const visibleSlice = events.slice(0, loadedCount);

  return (
    <Slab
      id="per-event"
      number="03"
      title="Per-event"
      trailing={
        events.length > 0 ? `₱${totalKept.toLocaleString()} total` : undefined
      }
    >
      {events.length === 0 ? (
        <p className="font-sans text-base text-slate max-w-md">
          Per-event revenue will land here once your first photo sells.
        </p>
      ) : (
        <>
          <ul className="border-y border-line divide-y divide-line">
            {visibleSlice.map((event) => (
              <li key={event.id}>
                <PerEventRow event={event} totalKept={totalKept} />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={visibleSlice.length}
            total={events.length}
            increment={PAGE_SIZE.EARNINGS_INCREMENT}
            onLoadMore={() =>
              setLoadedCount((n) => n + PAGE_SIZE.EARNINGS_INCREMENT)
            }
          />
        </>
      )}
    </Slab>
  );
}

function PerEventRow({
  event,
  totalKept,
}: {
  event: PhotographerEventSummary;
  totalKept: number;
}) {
  const sharePct = totalKept > 0 ? (event.revenueKept / totalKept) * 100 : 0;
  return (
    <Link
      href={`${ROUTES.DASHBOARD_EVENTS}/${event.id}`}
      className="group block py-5 md:py-6 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-2 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum">
            {formatLongDate(event.date, true)}
          </p>
          <h3 className="font-display text-lg md:text-xl font-medium tracking-tight text-ink mt-2 truncate group-hover:text-ink-soft transition-colors">
            {event.name}
          </h3>
          <p className="font-sans text-sm text-slate mt-2 tnum">
            <span className="font-mono text-ink-soft">{event.salesCount}</span>
            {" sold · "}
            <span className="font-mono text-ink-soft">{event.photoCount}</span>
            {" photos"}
          </p>
        </div>
        <div className="flex items-center gap-4 md:flex-col md:items-end md:gap-1 shrink-0">
          <p className="font-mono tnum font-medium text-ink text-lg md:text-xl">
            ₱{event.revenueKept.toLocaleString()}
          </p>
          <div className="flex items-center gap-2 md:gap-3">
            <div className="w-16 md:w-24 h-1 bg-bone-deep rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full bg-ink-soft",
                  "transition-[width]",
                )}
                style={{ width: `${Math.max(2, sharePct)}%` }}
              />
            </div>
            <p className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum uppercase shrink-0">
              {sharePct.toFixed(0)}%
            </p>
          </div>
        </div>
      </div>
    </Link>
  );
}
