"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useAdminTopPhotographersQuery } from "@/hooks/use-admin-data";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { photographerHref } from "@/lib/admin-photographer-view";
import { cn, formatPrice } from "@/lib/utils";
import { useTopEvents, type SalesEventRow } from "@/lib/admin-sales";
import type { AdminTopPhotographer } from "@/lib/api-admin";

const TOP_LIMIT = 10;

// Two-column ranked board for /admin/sales. Photographer side is the server's
// year-to-date list (GET /admin/sales/top-photographers, same sums as the
// money tiles). Event side is the server's order sums; a row derived in the
// browser while the server has no answer carries an Estimate badge.
export function AdminSalesLeaderboard() {
  const photographersQuery = useAdminTopPhotographersQuery("ytd", TOP_LIMIT);
  const photographers = photographersQuery.data;
  const events = useTopEvents(TOP_LIMIT);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-2 gap-10 md:gap-14">
      <Column
        title="Top photographers"
        caption="Year to date"
        emptyCopy="No photo sales yet this year."
        rows={photographers ?? []}
        fallback={
          photographers ? undefined : photographersQuery.isError ? (
            <div role="alert" className="space-y-3">
              <p className="font-sans text-sm text-error">
                Couldn&apos;t load the top photographers. Please try again.
              </p>
              <button
                type="button"
                onClick={() => void photographersQuery.refetch()}
                disabled={photographersQuery.isFetching}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                Retry
              </button>
            </div>
          ) : (
            <div aria-busy="true">
              <Skeleton className="h-40 w-full rounded-md" />
            </div>
          )
        }
        renderRow={(p, rank) => (
          <PhotographerRow key={p.userId} rank={rank} row={p} />
        )}
      />
      <Column
        title="Top events"
        caption="By total sales · all time"
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
  /** Shown instead of the list while it loads or after it fails. */
  fallback?: React.ReactNode;
  renderRow: (row: T, rank: number) => React.ReactNode;
}

function Column<T>({
  title,
  caption,
  emptyCopy,
  rows,
  fallback,
  renderRow,
}: ColumnProps<T>) {
  return (
    <div>
      <header className="flex items-baseline justify-between gap-3 mb-5">
        <Kicker>{title}</Kicker>
        <Kicker tone="soft">{caption}</Kicker>
      </header>
      {fallback ?? (rows.length === 0 ? (
        <p className="font-sans text-sm text-ink-soft">{emptyCopy}</p>
      ) : (
        <ol className="border-y border-line divide-y divide-line">
          {rows.map((row, i) => renderRow(row, i + 1))}
        </ol>
      ))}
    </div>
  );
}

function PhotographerRow({
  rank,
  row,
}: {
  rank: number;
  row: AdminTopPhotographer;
}) {
  const href = photographerHref(row);
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
            <p className="font-display text-base text-ink break-words group-hover:text-fresh transition-colors">
              {row.displayName}
            </p>
            <Kicker as="p" tone="soft" tnum className="mt-1">
              {row.salesCount.toLocaleString()}{" "}
              {row.salesCount === 1 ? "photo sold" : "photos sold"}
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
  const href = `${ROUTES.ADMIN_EVENTS}?q=${encodeURIComponent(row.name)}`;
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
              <p className="font-display text-base text-ink break-words group-hover:text-fresh transition-colors">
                {row.name}
              </p>
              {row.implied && (
                <Kicker tone="soft" className="border border-line rounded-full px-2 py-0.5">
                  Estimate
                </Kicker>
              )}
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
