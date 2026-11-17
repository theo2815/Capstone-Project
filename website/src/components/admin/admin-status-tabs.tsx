"use client";

import { cn } from "@/lib/utils";

export interface AdminStatusTab<K extends string> {
  key: K;
  label: string;
  count: number;
}

// Status tabs for admin lists (Payouts, Events): one status shows at a time,
// so an empty status costs one tab instead of a whole section.
export function AdminStatusTabs<K extends string>({
  label,
  tabs,
  value,
  onChange,
}: {
  label: string;
  tabs: ReadonlyArray<AdminStatusTab<K>>;
  value: K;
  onChange: (next: K) => void;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex flex-wrap gap-1 border-b border-line mt-8 md:mt-10">
      {tabs.map((t) => {
        const active = t.key === value;
        return (
          <button
            key={t.key}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(t.key)}
            className={cn(
              "-mb-px border-b-2 px-3.5 min-h-[44px] font-sans text-sm transition-colors",
              active ? "border-ink text-ink font-semibold" : "border-transparent text-slate hover:text-ink",
            )}
          >
            {t.label}
            <span className="ml-1.5 font-mono text-[12px] text-slate-soft tnum">{t.count}</span>
          </button>
        );
      })}
    </div>
  );
}
