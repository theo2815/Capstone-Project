"use client";

import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { AdminStatTile } from "@/components/admin/admin-stat-tile";
import { AdminSalesTrend } from "@/components/admin/admin-sales-trend";
import { AdminSalesLeaderboard } from "@/components/admin/admin-sales-leaderboard";
import { AdminSalesEventTable } from "@/components/admin/admin-sales-event-table";
import { ROUTES } from "@/lib/constants";
import {
  useSalesKpis,
  useRefundPulse,
} from "@/lib/admin-sales";
import { formatPrice } from "@/lib/utils";

// /admin/sales — revenue intelligence. Five slabs, all derivations sourced
// from existing mock seeds:
//   01 KPIs            ADMIN_PAYOUT_SEED + disputes
//   02 Weekly trend    ADMIN_PAYOUT_SEED grouped by weekOf
//   03 Top performers  same, plus EVENT_CATALOG (implied GMV for events)
//   04 Sales by event  EVENT_CATALOG × per-photo price + dispute join
//   05 Refund pulse    disputes with refundAmount > 0
//
// Per-photo price + cut rate live in lib/platform-economics.ts so this page
// and the photographer-facing PlatformCutModal stay in lockstep.

export default function AdminSalesPage() {
  const kpis = useSalesKpis();
  const refunds = useRefundPulse();

  return (
    <>
      <Header totalSales={kpis.totalSalesCount} />

      <Slab
        id="kpis"
        number="01"
        title="Money"
        caption="GMV and where it splits"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
          <AdminStatTile
            number="01"
            kicker="Gross merch volume"
            value={formatPrice(kpis.gmv)}
            caption={`${kpis.totalSalesCount.toLocaleString()} ${
              kpis.totalSalesCount === 1 ? "photo sold" : "photos sold"
            }`}
          />
          <AdminStatTile
            number="02"
            kicker="Platform revenue"
            value={formatPrice(kpis.platformRevenue)}
            caption="25% of every sale"
          />
          <AdminStatTile
            number="03"
            kicker="Refunds issued"
            value={formatPrice(kpis.refundsIssued)}
            caption={
              refunds.refundCount === 0
                ? "no refunds processed"
                : `${refunds.refundCount} ${
                    refunds.refundCount === 1 ? "refund" : "refunds"
                  } processed`
            }
          />
          <AdminStatTile
            number="04"
            kicker="Net platform revenue"
            value={formatPrice(kpis.netPlatformRevenue)}
            caption="cut minus refunds"
          />
        </div>
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
        <RefundPulseSummary />
      </Slab>
    </>
  );
}

function Header({ totalSales }: { totalSales: number }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker tone="soft">Sales · Revenue intelligence</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        The money.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Where revenue comes from across photographers, events, and the last
        twelve weeks. Event-side figures are{" "}
        <span className="text-ink">implied</span> until order-level
        attribution lands —{" "}
        <span className="font-mono tnum">{totalSales.toLocaleString()}</span>{" "}
        photos sold to date.
      </p>
    </header>
  );
}

function RefundPulseSummary() {
  const refunds = useRefundPulse();
  return (
    <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 md:gap-10 items-start">
      <ul className="border-y border-line divide-y divide-line">
        <PulseRow
          label="Refunds issued"
          value={formatPrice(refunds.refundsIssued)}
          caption={`${refunds.refundCount} ${
            refunds.refundCount === 1 ? "refund" : "refunds"
          } processed`}
        />
        <PulseRow
          label="Refund rate"
          value={`${refunds.refundRatePct.toFixed(2)}%`}
          caption="of gross merch volume"
        />
        <PulseRow
          label="Open · escalated"
          value={`${refunds.openCount} · ${refunds.escalatedCount}`}
          caption="awaiting decision"
        />
      </ul>

      <Link
        href={ROUTES.ADMIN_DISPUTES}
        className="inline-flex items-center gap-2 self-start font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors"
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
