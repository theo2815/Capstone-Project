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
import { SALES_TREND_WEEKS, useRefundPulse } from "@/lib/admin-sales";
import { PLATFORM_CUT_RATE } from "@/lib/platform-economics";
import { cn, formatPrice } from "@/lib/utils";

// /admin/sales — revenue intelligence. Five slabs:
//   01 KPIs            server KPIs only (`range=ytd`); never a browser estimate
//   02 Weekly trend    server weekly series (`/admin/sales/trend`)
//   03 Top performers  server top photographers (year to date) and event order sums
//   04 Sales by event  server order sums; browser-derived rows badged Implied
//   05 Refund pulse    server refunds; rate against `range=week` GMV
//
// Per-photo price + cut rate live in lib/platform-economics.ts so this page
// and the photographer-facing PlatformCutModal stay in lockstep.

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
        caption="GMV and where it splits"
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
            {Array.from({ length: 4 }, (_, i) => (
              <Skeleton key={i} className="h-40 w-full rounded-2xl" />
            ))}
          </div>
        ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 md:gap-8">
          <AdminStatTile
            kicker="Gross merch volume"
            value={formatPrice(kpis.gmv)}
            caption={`${kpis.totalSalesCount.toLocaleString()} ${
              kpis.totalSalesCount === 1 ? "photo sold" : "photos sold"
            }`}
          />
          <AdminStatTile
            kicker="Platform revenue"
            value={formatPrice(kpis.platformRevenue)}
            caption={`${Math.round(PLATFORM_CUT_RATE * 100)}% of every sale`}
          />
          <AdminStatTile
            kicker="Refunds issued"
            value={formatPrice(kpis.refundsIssued)}
            caption={refundCaption(kpis.refundsIssued, refunds.refundCount)}
          />
          <AdminStatTile
            kicker="Net platform revenue"
            value={formatPrice(kpis.netPlatformRevenue)}
            caption="cut minus refunds"
          />
        </div>
        )}
      </Slab>

      <Slab
        id="trend"
        number="02"
        title="Trend"
        caption="GMV per week"
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
        caption="Sortable · implied figures flagged"
      >
        <AdminSalesEventTable />
      </Slab>

      <Slab
        id="refunds"
        number="05"
        title="Refund pulse"
        caption="Pressure on net revenue"
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
      <Kicker tone="soft">Sales · Revenue intelligence</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        The money.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Where revenue comes from across photographers, events, and the last{" "}
        <span className="font-mono tnum">{SALES_TREND_WEEKS}</span> weeks —{" "}
        <span className="font-mono tnum">{totalSales?.toLocaleString() ?? "—"}</span>{" "}
        photos sold to date.
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
    <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 md:gap-10 items-start">
      <ul className="border-y border-line divide-y divide-line">
        <PulseRow
          label="Refunds issued"
          value={refundsIssued === undefined ? "—" : formatPrice(refundsIssued)}
          caption={refundsIssued === undefined ? "not loaded" : `${refundCaption(refundsIssued, refunds.refundCount)}, year to date`}
        />
        <PulseRow
          label="Refund rate"
          value={rate === null ? "—" : `${rate.toFixed(2)}%`}
          caption="of this week's gross merch volume"
        />
        <PulseRow
          label="Open · escalated"
          value={`${refunds.openCount} · ${refunds.escalatedCount}`}
          caption="awaiting decision"
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
