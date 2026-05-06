"use client";

import { useState } from "react";
import { formatLongDate, formatMonthYear } from "@/lib/format";
import { cn } from "@/lib/utils";

interface SparklinePoint {
  weekOf: string;
  amount: number;
}

interface SparklineProps {
  data: ReadonlyArray<SparklinePoint>;
  ariaLabel?: string;
  interactive?: boolean;
}

// 12-week earnings sparkline. ink-soft bars on a bone-deep track. Stays out
// of the fresh accent so the lifetime ₱ above can own it.
//
// `interactive` opts in to hover/tap-to-pin: bars become focusable buttons,
// the active bar deepens to ink, and a label row below the chart morphs
// between axis labels (default) and the active week's readout.
export function Sparkline({
  data,
  ariaLabel,
  interactive = false,
}: SparklineProps) {
  const max = Math.max(...data.map((d) => d.amount), 1);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);

  if (!interactive) {
    return (
      <div
        role="img"
        aria-label={ariaLabel}
        className="flex items-end gap-1 h-16 md:h-20 bg-bone-deep rounded-md p-1.5"
      >
        {data.map((point, i) => {
          const pct = Math.max(4, (point.amount / max) * 100);
          return (
            <div
              key={point.weekOf}
              style={{
                height: `${pct}%`,
                animation: `fade-in 0.5s ${i * 0.04}s both`,
                opacity: 0,
              }}
              className="flex-1 bg-ink-soft rounded-sm"
            />
          );
        })}
      </div>
    );
  }

  const active = activeIndex !== null ? data[activeIndex] : null;
  const firstWeek = data[0]?.weekOf;
  const lastWeek = data[data.length - 1]?.weekOf;

  return (
    <div>
      <div
        role="img"
        aria-label={ariaLabel}
        className="flex items-end gap-1 h-16 md:h-20 bg-bone-deep rounded-md p-1.5"
        onMouseLeave={() => setActiveIndex(null)}
      >
        {data.map((point, i) => {
          const pct = Math.max(4, (point.amount / max) * 100);
          const isActive = activeIndex === i;
          return (
            <button
              key={point.weekOf}
              type="button"
              onMouseEnter={() => setActiveIndex(i)}
              onFocus={() => setActiveIndex(i)}
              onClick={() =>
                setActiveIndex(activeIndex === i ? null : i)
              }
              aria-label={`Week of ${formatLongDate(point.weekOf)}: ₱${point.amount.toLocaleString()}`}
              className="flex-1 h-full flex items-end rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-fresh focus-visible:ring-offset-1 focus-visible:ring-offset-bone-deep cursor-pointer"
            >
              <span
                style={{
                  height: `${pct}%`,
                  animation: `fade-in 0.5s ${i * 0.04}s both`,
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
            <span>Week of {formatLongDate(active.weekOf, true)}</span>
            <span className="text-slate-soft mx-2">·</span>
            <span className="text-ink">
              ₱{active.amount.toLocaleString()}
            </span>
          </p>
        ) : (
          firstWeek &&
          lastWeek && (
            <div className="flex-1 flex items-center justify-between text-slate-soft">
              <span>{formatMonthYear(firstWeek)}</span>
              <span>{formatMonthYear(lastWeek)}</span>
            </div>
          )
        )}
      </div>
    </div>
  );
}
