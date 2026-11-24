"use client";

import { useMemo } from "react";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useAdminSalesTrendQuery } from "@/hooks/use-admin-data";
import { SALES_TREND_WEEKS } from "@/lib/admin-sales";
import { cn, formatPrice } from "@/lib/utils";
import { AdminTrendChart } from "./admin-trend-chart";

// 12-week total-sales bars for /admin/sales, from GET /admin/sales/trend. Wraps
// <AdminTrendChart> with weekly label formatting + peso readout so the same
// visual idiom serves both daily activity (Overview) and weekly money (Sales).
const WEEK_FORMATTER = new Intl.DateTimeFormat("en-PH", {
  month: "short",
  day: "2-digit",
});

function formatWeekLabel(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return `Wk of ${WEEK_FORMATTER.format(d)}`.toUpperCase();
}

export function AdminSalesTrend() {
  const trendQuery = useAdminSalesTrendQuery(SALES_TREND_WEEKS);
  const series = trendQuery.data;
  const data = useMemo(
    () => (series ?? []).map((p) => ({ date: p.weekStart, amount: p.gmv })),
    [series],
  );
  const total = data.reduce((sum, p) => sum + p.amount, 0);
  // Same all-zero check as the Overview trend: a sentence, not an empty grey chart.
  const noSales = !!series?.every((p) => p.gmv === 0);

  if (!series && trendQuery.isError) {
    return (
      <div role="alert" className="space-y-3">
        <p className="font-sans text-sm text-error">
          Couldn&apos;t load the weekly trend. Please try again.
        </p>
        <button
          type="button"
          onClick={() => void trendQuery.refetch()}
          disabled={trendQuery.isFetching}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Retry
        </button>
      </div>
    );
  }
  if (!series) {
    return (
      <div aria-busy="true">
        <Skeleton className="h-20 md:h-24 w-full rounded-md" />
      </div>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-5">
        <Kicker>Weekly total sales · last {series.length} weeks</Kicker>
        <Kicker tone="soft" tnum>
          {formatPrice(total)} total
        </Kicker>
      </div>

      {noSales ? (
        <p className="font-sans text-sm text-ink-soft">
          No sales in the last {series.length} weeks.
        </p>
      ) : (
        <AdminTrendChart
          data={data}
          unitLabel="total sales"
          ariaLabel={`Weekly total sales for the last ${series.length} weeks`}
          formatLabel={formatWeekLabel}
          formatValue={formatPrice}
        />
      )}
    </div>
  );
}
