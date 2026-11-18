"use client";

import { useState } from "react";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";

export interface DailyMetric {
  date: string; // YYYY-MM-DD
  amount: number;
}

interface AdminTrendChartProps {
  data: ReadonlyArray<DailyMetric>;
  /** Used in the active-bar readout: "{n} {unit} on {date}" */
  unitLabel: string;
  /** Caption of the screen-reader table that stands in for the bars. */
  ariaLabel: string;
  /** Override how each `point.date` renders in the readout + aria. Default
   *  renders the long-form day (e.g. "APR 28 · 2026"). */
  formatLabel?: (date: string) => string;
  /** Override how each `point.amount` renders in the readout. Default joins
   *  toLocaleString() with `unitLabel`. */
  formatValue?: (amount: number) => string;
}

// Bar chart for /admin/* trend slabs. Originally written for the Overview's
// 30-day daily series; the optional `formatLabel` + `formatValue` props let
// the same component serve weekly currency series on /admin/sales. The
// visual idiom (bone-deep track + ink-soft bars + ink active) stays
// identical so admin trend slabs feel cohesive across pages.
export function AdminTrendChart({
  data,
  unitLabel,
  ariaLabel,
  formatLabel,
  formatValue,
}: AdminTrendChartProps) {
  const max = Math.max(...data.map((d) => d.amount), 1);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const active = activeIndex !== null ? data[activeIndex] ?? null : null;
  const first = data[0];
  const last = data[data.length - 1];

  const renderLabel = (date: string) =>
    formatLabel ? formatLabel(date) : formatLongDate(date, true);
  const renderValue = (amount: number) =>
    formatValue
      ? formatValue(amount)
      : `${amount.toLocaleString()} ${unitLabel}`;

  return (
    <div>
      <table className="sr-only">
        <caption>{ariaLabel}</caption>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">{unitLabel}</th>
          </tr>
        </thead>
        <tbody>
          {data.map((point) => (
            <tr key={point.date}>
              <td>{renderLabel(point.date)}</td>
              <td>{renderValue(point.amount)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {/* Bars are for the eye only; the table above is their text version. */}
      <div
        aria-hidden="true"
        className="flex items-end gap-1 h-20 md:h-24 bg-bone-deep rounded-md p-1.5"
        onMouseLeave={() => setActiveIndex(null)}
      >
        {data.map((point, i) => {
          const pct = (point.amount / max) * 100;
          const isActive = activeIndex === i;
          return (
            <span
              key={point.date}
              onMouseEnter={() => setActiveIndex(i)}
              className="flex-1 h-full flex items-end rounded-sm"
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
            </span>
          );
        })}
      </div>

      <div className="mt-3 h-4 flex items-center font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.15em] uppercase tnum">
        {active ? (
          <p className="flex-1 text-center text-slate">
            <span>{renderLabel(active.date)}</span>
            <span className="text-slate-soft mx-2">·</span>
            <span className="text-ink">{renderValue(active.amount)}</span>
          </p>
        ) : (
          first &&
          last && (
            <div className="flex-1 flex items-center justify-between text-slate-soft">
              <span>{renderLabel(first.date)}</span>
              <span>{renderLabel(last.date)}</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}
