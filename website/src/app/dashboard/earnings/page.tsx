"use client";

import { useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { Sparkline } from "@/components/dashboard/sparkline";
import { PlatformCutModal } from "@/components/dashboard/platform-cut-modal";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import {
  PHOTOGRAPHER_EARNINGS,
  PHOTOGRAPHER_EVENTS,
  type PhotographerEventSummary,
} from "@/lib/photographer-mock";
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
  const e = PHOTOGRAPHER_EARNINGS;
  const [cutModalOpen, setCutModalOpen] = useState(false);

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
  const e = PHOTOGRAPHER_EARNINGS;
  return (
    <Slab id="breakdown" number="02" title="Breakdown" caption="Current cycle">
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
      <p className="font-sans text-xs md:text-sm text-slate mt-3">{caption}</p>
    </div>
  );
}

function PerEventSlab() {
  // Sort events by lifetime revenue kept, descending. Skip events with zero
  // (upcoming events not yet covered) so the table shows revenue-bearing
  // entries only. The full list lives at /dashboard/events.
  const events = [...PHOTOGRAPHER_EVENTS]
    .filter((e) => e.revenueKept > 0)
    .sort((a, b) => b.revenueKept - a.revenueKept);

  const totalKept = events.reduce((sum, e) => sum + e.revenueKept, 0);

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
        <ul className="border-y border-line divide-y divide-line">
          {events.map((event) => (
            <li key={event.id}>
              <PerEventRow event={event} totalKept={totalKept} />
            </li>
          ))}
        </ul>
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
