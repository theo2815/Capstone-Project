"use client";

import type { EventState } from "@/app/events/events-browser";
import {
  Dropdown,
  DropdownItem,
  DropdownTrigger,
} from "@/components/ui/dropdown";
import { Kicker } from "@/components/ui/kicker";

// Shared event filter bar for photographer surfaces. Originally lived inline
// in /profile (PortfolioFilterBar). Lifted in Slice 2 of the edge-case
// hardening pass so /dashboard/events and /dashboard/upload inherit the same
// "search by name + filter by date" pattern instead of each surface rolling
// its own.
//
// Date taxonomy mirrors /events so the photographer's "what runners see"
// mental model lines up across the dashboard. matchState maps the dashboard
// label back to the canonical EventState the catalog stores.

export type EventDateKey =
  | "any"
  | "upcoming"
  | "live"
  | "recent"
  | "archive";

interface EventDateOption {
  value: EventDateKey;
  label: string;
  matchState?: EventState;
}

export const EVENT_DATE_OPTIONS: ReadonlyArray<EventDateOption> = [
  { value: "any", label: "Any time" },
  { value: "upcoming", label: "Upcoming", matchState: "upcoming" },
  { value: "live", label: "Live now", matchState: "live" },
  { value: "recent", label: "Recent", matchState: "open" },
  { value: "archive", label: "Archive", matchState: "past" },
];

// Predicate helper so consumers don't re-derive the matchState lookup. Returns
// true when the event's state matches the selected date key, OR when "any"
// is selected (no date filter active).
export function matchEventDate(
  state: EventState,
  dateKey: EventDateKey,
): boolean {
  if (dateKey === "any") return true;
  const want = EVENT_DATE_OPTIONS.find((d) => d.value === dateKey)?.matchState;
  return want === undefined || state === want;
}

interface EventFilterBarProps {
  date: EventDateKey;
  onDateChange: (next: EventDateKey) => void;
  query: string;
  onQueryChange: (next: string) => void;
  /** Optional aria-label override on the date dropdown — defaults to the
   *  /profile copy. Pass a context-specific label on dashboard surfaces. */
  dateAriaLabel?: string;
  /** Optional placeholder override for the search input. */
  searchPlaceholder?: string;
  /** Optional aria-label override for the search input. */
  searchAriaLabel?: string;
}

export function EventFilterBar({
  date,
  onDateChange,
  query,
  onQueryChange,
  dateAriaLabel = "Filter events by date",
  searchPlaceholder = "Search events…",
  searchAriaLabel = "Search events",
}: EventFilterBarProps) {
  const dateLabel =
    EVENT_DATE_OPTIONS.find((d) => d.value === date)?.label ?? "Any time";
  return (
    <div className="mb-6 md:mb-8 flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-6 border-b border-line pb-4">
      <div className="sm:flex-1 min-w-0">
        <EventSearchInput
          value={query}
          onChange={onQueryChange}
          placeholder={searchPlaceholder}
          ariaLabel={searchAriaLabel}
        />
      </div>
      <Dropdown
        ariaLabel={dateAriaLabel}
        align="right"
        trigger={<DropdownTrigger>{dateLabel}</DropdownTrigger>}
      >
        {EVENT_DATE_OPTIONS.map((d) => (
          <DropdownItem
            key={d.value}
            variant="mono"
            active={d.value === date}
            onClick={() => onDateChange(d.value)}
          >
            {d.label}
          </DropdownItem>
        ))}
      </Dropdown>
    </div>
  );
}

function EventSearchInput({
  value,
  onChange,
  placeholder,
  ariaLabel,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  ariaLabel: string;
}) {
  return (
    <div className="relative">
      <svg
        className="absolute left-0 top-1/2 -translate-y-1/2 size-4 text-slate pointer-events-none"
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <circle
          cx="11"
          cy="11"
          r="7"
          stroke="currentColor"
          strokeWidth="1.5"
        />
        <path
          d="M16.5 16.5 L21 21"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label={ariaLabel}
        className="block w-full pl-7 pr-2 py-1.5 bg-transparent border-0 outline-none font-mono text-sm tracking-wide placeholder:text-slate-soft text-ink"
      />
    </div>
  );
}

// Empty state when filters yield zero matches. Lifted from /profile alongside
// the bar so dashboard surfaces get a consistent "Nothing matches · Clear
// filters" affordance rather than each page rolling its own copy.
export function EventFilterEmpty({ onClear }: { onClear: () => void }) {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <Kicker as="p" className="mb-3">
        No matches
      </Kicker>
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        Nothing here matches.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Try clearing the search or change the date filter.
      </p>
      <button
        type="button"
        onClick={onClear}
        className="mt-6 font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
      >
        Clear filters
      </button>
    </div>
  );
}
