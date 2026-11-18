"use client";

import { useMemo } from "react";
import { Kicker } from "@/components/ui/kicker";
import { SALES_ESTIMATE_NOTE, useWeeklyGmvSeries } from "@/lib/admin-sales";
import { formatPrice } from "@/lib/utils";
import { AdminTrendChart } from "./admin-trend-chart";

// 12-week GMV bars for /admin/sales. Wraps <AdminTrendChart> with weekly
// label formatting + peso readout so the same visual idiom serves both
// daily activity (Overview) and weekly money (Sales).
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
  const series = useWeeklyGmvSeries();
  const data = useMemo(
    () => series.map((p) => ({ date: p.weekOf, amount: p.gmv })),
    [series],
  );
  const total = useMemo(
    () => series.reduce((sum, p) => sum + p.gmv, 0),
    [series],
  );

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-2">
        <div className="flex items-center gap-3">
          <Kicker>Weekly GMV · last {series.length} weeks</Kicker>
          <Kicker tone="soft" className="border border-line rounded-full px-2 py-0.5">
            Estimate
          </Kicker>
        </div>
        <Kicker tone="soft" tnum>
          {formatPrice(total)} total
        </Kicker>
      </div>
      <p className="font-sans text-sm text-ink-soft mb-5">{SALES_ESTIMATE_NOTE}</p>

      <AdminTrendChart
        data={data}
        unitLabel="GMV"
        ariaLabel={`Weekly gross merchandise volume for the last ${series.length} weeks`}
        formatLabel={formatWeekLabel}
        formatValue={formatPrice}
      />
    </div>
  );
}
