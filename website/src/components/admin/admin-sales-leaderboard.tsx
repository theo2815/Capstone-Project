"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";
import {
  useTopPhotographers,
  useTopEvents,
  type TopPhotographer,
  type SalesEventRow,
} from "@/lib/admin-sales";

const TOP_LIMIT = 10;

// Two-column ranked board for /admin/sales. Photographer side is grounded in
// real cycle data (ADMIN_PAYOUT_SEED). Event side is "implied GMV" — derived
// from EVENT_CATALOG.photoCount × per-photo price — and carries an Implied
// kicker badge until backend Phase F lands order-level event attribution.
export function AdminSalesLeaderboard() {
  const photographers = useTopPhotographers(TOP_LIMIT);
  const events = useTopEvents(TOP_LIMIT);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-10 md:gap-14">
      <Column
        title="Top photographers"
        caption="By gross merchandise volume"
        emptyCopy="No payouts yet — once payouts land, the highest-earning brands surface here."
        rows={photographers}
        renderRow={(p, rank) => (
          <PhotographerRow key={p.photographerId} rank={rank} row={p} />
        )}
      />
      <Column
        title="Top events"
        caption="Implied · until order attribution lands"
        emptyCopy="No events with photo sales yet."
        rows={events}
        renderRow={(e, rank) => (
          <EventRow key={e.id} rank={rank} row={e} />
        )}
      />
    </div>
  );
}

interface ColumnProps<T> {
  title: string;
  caption: string;
  emptyCopy: string;
  rows: T[];
  renderRow: (row: T, rank: number) => React.ReactNode;
}

function Column<T>({
  title,
  caption,
  emptyCopy,
  rows,
  renderRow,
}: ColumnProps<T>) {
  return (
    <div>
      <header className="flex items-baseline justify-between gap-3 mb-5">
        <Kicker>{title}</Kicker>
        <Kicker tone="soft">{caption}</Kicker>
      </header>
      {rows.length === 0 ? (
        <p className="font-sans text-sm text-ink-soft">{emptyCopy}</p>
      ) : (
        <ol className="border-y border-line divide-y divide-line">
          {rows.map((row, i) => renderRow(row, i + 1))}
        </ol>
      )}
    </div>
  );
}

function PhotographerRow({
  rank,
  row,
}: {
  rank: number;
  row: TopPhotographer;
}) {
  const label = row.brandName ?? row.photographerName;
  const href = row.handle
    ? `${ROUTES.ADMIN_PHOTOGRAPHERS}/${row.handle}`
    : ROUTES.ADMIN_PHOTOGRAPHERS;
  return (
    <li>
      <Link
        href={href}
        className="group flex items-baseline justify-between gap-4 py-4 md:py-5 transition-colors"
      >
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono tnum text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft shrink-0">
            {rank.toString().padStart(2, "0")}
          </span>
          <div className="min-w-0">
            <p className="font-display text-base text-ink truncate group-hover:text-fresh transition-colors">
              {label}
            </p>
            <Kicker as="p" tone="soft" tnum className="mt-1">
              {row.photosSold.toLocaleString()} photos · {row.cycles}{" "}
              {row.cycles === 1 ? "payout" : "payouts"}
            </Kicker>
          </div>
        </div>
        <p className="font-mono tnum font-medium text-ink text-base md:text-lg shrink-0">
          {formatPrice(row.gmv)}
        </p>
      </Link>
    </li>
  );
}

function EventRow({ rank, row }: { rank: number; row: SalesEventRow }) {
  const href = `${ROUTES.ADMIN_EVENTS}/${row.id}`;
  return (
    <li>
      <Link
        href={href}
        className="group flex items-baseline justify-between gap-4 py-4 md:py-5 transition-colors"
      >
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono tnum text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft shrink-0">
            {rank.toString().padStart(2, "0")}
          </span>
          <div className="min-w-0">
            <div className="flex items-baseline gap-3 flex-wrap">
              <p className="font-display text-base text-ink truncate group-hover:text-fresh transition-colors">
                {row.name}
              </p>
              <Kicker tone="soft" className="border border-line rounded-full px-2 py-0.5">
                Implied
              </Kicker>
            </div>
            <Kicker as="p" tone="soft" tnum className="mt-1">
              {formatLongDate(row.date, true)} · {row.photoCount.toLocaleString()} photos
            </Kicker>
          </div>
        </div>
        <p className="font-mono tnum font-medium text-ink text-base md:text-lg shrink-0">
          {formatPrice(row.impliedGmv)}
        </p>
      </Link>
    </li>
  );
}
