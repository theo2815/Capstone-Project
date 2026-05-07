"use client";

import { useState } from "react";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { DailyMetric } from "@/lib/admin-overview-mock";

interface AdminTrendChartProps {
  data: ReadonlyArray<DailyMetric>;
  /** Used in the active-bar readout: "{n} {unit} on {date}" */
  unitLabel: string;
  /** aria-label for the whole chart. */
  ariaLabel: string;
}

// 30-day bar chart for the admin Overview. Sibling of <Sparkline> — kept
// separate because <Sparkline> is earnings-shaped (12 weeks, currency
// labels). This one takes a generic series and renders bone-deep track +
// ink-soft bars (active = ink). Endpoints labeled with first/last day.
// Hover/focus a bar to pin the readout below the chart.
export function AdminTrendChart({
  data,
  unitLabel,
  ariaLabel,
}: AdminTrendChartProps) {
  const max = Math.max(...data.map((d) => d.amount), 1);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const active = activeIndex !== null ? data[activeIndex] ?? null : null;
  const first = data[0];
  const last = data[data.length - 1];

  return (
    <div>
      <div
        role="img"
        aria-label={ariaLabel}
        className="flex items-end gap-1 h-20 md:h-24 bg-bone-deep rounded-md p-1.5"
        onMouseLeave={() => setActiveIndex(null)}
      >
        {data.map((point, i) => {
          const pct = Math.max(4, (point.amount / max) * 100);
          const isActive = activeIndex === i;
          return (
            <button
              key={point.date}
              type="button"
              onMouseEnter={() => setActiveIndex(i)}
              onFocus={() => setActiveIndex(i)}
              onClick={() =>
                setActiveIndex(activeIndex === i ? null : i)
              }
              aria-label={`${formatLongDate(point.date, true)}: ${point.amount.toLocaleString()} ${unitLabel}`}
              className="flex-1 h-full flex items-end rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-fresh focus-visible:ring-offset-1 focus-visible:ring-offset-bone-deep cursor-pointer"
            >
              <span
                style={{
                  height: `${pct}%`,
                  animation: `fade-in 0.5s ${i * 0.015}s both`,
                  opacity: 0,
                }}
                className={cn(
                  "block w-full rounded-sm transition-colors",
                  isActive ? "bg-ink" : "bg-ink-soft",
                )}
              />
            </button>
          );
        })}
      </div>

      <div className="mt-3 h-4 flex items-center font-mono text-[10px] tracking-[0.15em] uppercase tnum">
        {active ? (
          <p className="flex-1 text-center text-slate">
            <span>{formatLongDate(active.date, true)}</span>
            <span className="text-slate-soft mx-2">·</span>
            <span className="text-ink">
              {active.amount.toLocaleString()} {unitLabel}
            </span>
          </p>
        ) : (
          first &&
          last && (
            <div className="flex-1 flex items-center justify-between text-slate-soft">
              <span>{formatLongDate(first.date, true)}</span>
              <span>{formatLongDate(last.date, true)}</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}
