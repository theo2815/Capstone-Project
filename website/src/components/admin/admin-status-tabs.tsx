"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export interface AdminStatusTab<K extends string> {
  key: K;
  label: string;
  /** Left out for tabs that are not statuses (the Sales sort). */
  count?: number;
  /** The queue that needs attention first: the one breathing fresh dot (the Inbox). */
  dot?: boolean;
}

// Status tabs for admin lists (Payouts, Events): one status shows at a time,
// so an empty status costs one tab instead of a whole section.
export function AdminStatusTabs<K extends string>({
  label,
  tabs,
  value,
  onChange,
  scroll = false,
  countLabel,
  className,
}: {
  label: string;
  tabs: ReadonlyArray<AdminStatusTab<K>>;
  value: K;
  onChange: (next: K) => void;
  /** One row that scrolls sideways below md instead of wrapping (the Inbox chips). */
  scroll?: boolean;
  /** Spoken after each count, e.g. "waiting" names a chip "Disputes, 2 waiting". */
  countLabel?: string;
  /** Page-only spacing, e.g. the Inbox's tighter top margin. */
  className?: string;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  // A scrolled row centres the selected tab (clear of the fade at the right edge) when the tab
  // changes and once the counts arrive and widen the tabs. It scrolls the row only, never the page,
  // and later count refreshes leave the admin's own swipe alone.
  const countsKnown = tabs.some((t) => t.count !== undefined);
  useEffect(() => {
    const row = rowRef.current;
    const tab = row?.querySelector<HTMLElement>('[aria-pressed="true"]');
    if (!scroll || !row || !tab) return;
    const r = row.getBoundingClientRect();
    const t = tab.getBoundingClientRect();
    row.scrollLeft += t.left - r.left - (r.width - t.width) / 2;
  }, [scroll, value, countsKnown]);

  return (
    <div
      ref={rowRef}
      role="group"
      aria-label={label}
      className={cn(
        "flex gap-1 border-b border-line mt-8 md:mt-10",
        scroll
          ? "flex-nowrap md:flex-wrap overflow-x-auto md:overflow-visible pb-px md:pb-0 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden [mask-image:linear-gradient(to_right,black_85%,transparent)] md:[mask-image:none] after:content-[''] after:w-[15%] after:shrink-0 md:after:hidden"
          : "flex-wrap",
        className,
      )}
    >
      {tabs.map((t) => {
        const active = t.key === value;
        const name =
          countLabel || t.dot
            ? [
                t.label,
                t.count !== undefined && (countLabel ? `${t.count} ${countLabel}` : `${t.count}`),
                t.dot && "needs attention",
              ]
                .filter(Boolean)
                .join(", ")
            : undefined;
        return (
          <button
            key={t.key}
            type="button"
            aria-pressed={active}
            aria-label={name}
            onClick={() => onChange(t.key)}
            className={cn(
              "-mb-px border-b-2 px-3.5 min-h-[44px] font-sans text-sm transition-colors",
              // The scroll row clips overflow, so the focus ring sits inside the tab.
              scroll && "shrink-0 whitespace-nowrap focus-visible:outline-offset-[-2px]",
              active ? "border-ink text-ink font-semibold" : "border-transparent text-slate hover:text-ink",
            )}
          >
            {t.label}
            {t.count !== undefined && (
              <span className="ml-1.5 font-mono text-[12px] text-slate-soft tnum">{t.count}</span>
            )}
            {t.dot && (
              <span aria-hidden="true" className="ml-1.5 inline-block align-middle size-1.5 rounded-full bg-fresh breathe" />
            )}
          </button>
        );
      })}
    </div>
  );
}
