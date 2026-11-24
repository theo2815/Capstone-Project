"use client";

import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { AdminStatTile } from "@/components/admin/admin-stat-tile";
import { AdminSalesTrend } from "@/components/admin/admin-sales-trend";
import { AdminSalesLeaderboard } from "@/components/admin/admin-sales-leaderboard";
import { AdminSalesEventTable } from "@/components/admin/admin-sales-event-table";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";
import { useAdminSalesKpisQuery } from "@/hooks/use-admin-data";
import { useRefundPulse } from "@/lib/admin-sales";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { cn, formatPrice } from "@/lib/utils";

// /admin/sales — revenue intelligence. Five slabs:
//   01 KPIs            server KPIs only (`range=ytd`); never a browser estimate
//   02 Weekly trend    server weekly series (`/admin/sales/trend`)
//   03 Top performers  server top photographers (year to date) and event order sums
//   04 Sales by event  server order sums; browser-derived rows badged Estimate
//   05 Refund pulse    server refunds; rate against `range=week` GMV

export default function AdminSalesPage() {
  const kpisQuery = useAdminSalesKpisQuery("ytd");
  const kpis = kpisQuery.data;
  const refunds = useRefundPulse();

  return (
    <>
      <Header totalSales={kpis?.totalSalesCount} />

      <Slab
        id="kpis"
        number="01"
        title="Money"
        caption="Year to date"
      >
        {!kpis && kpisQuery.isError ? (
          <div role="alert" className="space-y-3">
            <p className="font-sans text-sm text-error">
              Couldn&apos;t load the sales figures. Please try again.
            </p>
            <button
              type="button"
              onClick={() => void kpisQuery.refetch()}
              disabled={kpisQuery.isFetching}
              className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
            >
              Retry
            </button>
          </div>
        ) : !kpis ? (
          <div
            className="grid grid-cols-1 xl:grid-cols-2 gap-6 md:gap-8"
            aria-busy="true"
          >
            {Array.from({ length: 3 }, (_, i) => (
              <Skeleton key={i} className="h-40 w-full rounded-2xl" />
            ))}
          </div>
        ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 md:gap-8">
          <AdminStatTile
            kicker="Total sales"
            value={formatPrice(kpis.gmv)}
            caption={`${kpis.totalSalesCount.toLocaleString()} ${
              kpis.totalSalesCount === 1 ? "photo sold" : "photos sold"
            } · year to date`}
          />
          {/* One tile: the backend's netPlatformRevenue equals platformRevenue, refunds already taken out. */}
          <AdminStatTile
            kicker="Platform cut"
            value={formatPrice(kpis.platformRevenue)}
            caption="QuickPitik's share after refunds · year to date"
          />
          <AdminStatTile
            kicker="Refunds issued"
            value={formatPrice(kpis.refundsIssued)}
            caption={`${refundCaption(kpis.refundsIssued, refunds.refundCount)} · year to date`}
          />
        </div>
        )}
      </Slab>

      <Slab
        id="trend"
        number="02"
        title="Trend"
        caption="Total sales per week"
      >
        <AdminSalesTrend />
      </Slab>

      <Slab
        id="performers"
        number="03"
        title="Top performers"
        caption="Photographers and events"
      >
        <AdminSalesLeaderboard />
      </Slab>

      <Slab
        id="by-event"
        number="04"
        title="Sales by event"
        caption="Sortable · estimates are marked"
      >
        <AdminSalesEventTable />
      </Slab>

      <Slab
        id="refunds"
        number="05"
        title="Refund pulse"
        caption="This week and year to date"
      >
        <RefundPulseSummary refundsIssued={kpis?.refundsIssued} />
      </Slab>
    </>
  );
}

// The amount is the server total and the count comes from the dispute list, so
// never caption a non-zero amount with "no refunds" or "0 refunds".
function refundCaption(amount: number, count: number): string {
  if (amount === 0) return "no refunds processed";
  if (count === 0) return "refunded to runners";
  return `${count} ${count === 1 ? "refund" : "refunds"} processed`;
}

function Header({ totalSales }: { totalSales: number | undefined }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker>Sales · Money report</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Sales.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_SALES} />
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        <span className="font-mono tnum">{totalSales?.toLocaleString() ?? "—"}</span>{" "}
        photos sold so far this year.
      </p>
    </header>
  );
}

function RefundPulseSummary({ refundsIssued }: { refundsIssued: number | undefined }) {
  const refunds = useRefundPulse();
  const week = useAdminSalesKpisQuery("week").data;
  // No sales this week means no rate (a refund can be for an older sale).
  const rate = week && week.gmv > 0 ? (week.refundsIssued / week.gmv) * 100 : null;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-6 lg:gap-10 items-start">
      <ul className="border-y border-line divide-y divide-line">
        <PulseRow
          label="Refunds issued"
          value={refundsIssued === undefined ? "—" : formatPrice(refundsIssued)}
          caption={refundsIssued === undefined ? "not loaded" : `${refundCaption(refundsIssued, refunds.refundCount)}, year to date`}
        />
        <PulseRow
          label="Refund rate"
          value={rate === null ? "—" : `${rate.toFixed(2)}%`}
          caption="of this week's total sales"
        />
        <PulseRow
          label="Disputes still open"
          value={`${refunds.openCount} open, ${refunds.escalatedCount} escalated`}
          caption="waiting for a decision"
        />
      </ul>

      <Link
        href={ROUTES.ADMIN_DISPUTES}
        className={`inline-flex items-center gap-2 self-start font-mono uppercase ${KICKER_SIZE_CLASS.sm} text-ink hover:text-fresh transition-colors`}
      >
        Open disputes queue →
      </Link>
    </div>
  );
}

function PulseRow({
  label,
  value,
  caption,
}: {
  label: string;
  value: string;
  caption: string;
}) {
  return (
    <li className="py-4 md:py-5 flex items-baseline justify-between gap-4">
      <div>
        <Kicker as="p" tone="soft">
          {label}
        </Kicker>
        <p className="font-sans text-sm text-ink-soft mt-1">{caption}</p>
      </div>
      <p className="font-mono tnum font-medium text-ink text-base md:text-lg shrink-0">
        {value}
      </p>
    </li>
  );
}
