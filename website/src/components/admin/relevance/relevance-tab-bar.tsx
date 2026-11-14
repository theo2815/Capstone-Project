"use client";

import { RELEVANCE_TABS, type RelevanceTab } from "@/lib/admin-relevance";
import { cn } from "@/lib/utils";

/** The four Photo Relevance tabs; the host keeps the value in the URL (`?tab=`). */
export function RelevanceTabBar({ tab, onChange }: { tab: RelevanceTab; onChange: (next: RelevanceTab) => void }) {
  return (
    <div role="tablist" aria-label="Photo Relevance views" className="flex flex-wrap gap-2">
      {RELEVANCE_TABS.map((t) => {
        const active = tab === t.value;
        return (
          <button
            key={t.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(t.value)}
            className={cn(
              "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-4 py-2 min-h-[44px] transition-colors",
              active
                ? "bg-ink text-surface border border-ink"
                : "bg-bone text-slate border border-line hover:text-ink hover:border-ink",
            )}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
