"use client";

import { useEffect, useMemo, useState } from "react";
import type { Event } from "@/types/event";
import {
  Dropdown,
  DropdownItem,
  DropdownTrigger,
} from "@/components/ui/dropdown";
import { EventTile } from "@/components/events/event-tile";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { useEventCatalog } from "@/lib/event-catalog";
import { useUrlState, useUrlStateBatch } from "@/hooks/use-url-state";
import { PAGE_SIZE } from "@/lib/pagination-config";

export type EventState = "upcoming" | "live" | "open" | "past";
export type ListEvent = Event & { state: EventState; city: string };

type DateKey = "any" | "upcoming" | "live" | "recent" | "archive";

interface EventsBrowserProps {
  events: ReadonlyArray<ListEvent>;
}

const DATE_OPTIONS: {
  value: DateKey;
  label: string;
  matchState?: EventState;
}[] = [
  { value: "any", label: "Any time" },
  { value: "upcoming", label: "Upcoming", matchState: "upcoming" },
  { value: "live", label: "Live now", matchState: "live" },
  { value: "recent", label: "Recent", matchState: "open" },
  { value: "archive", label: "Archive", matchState: "past" },
];

const SEGMENT_GROUPS: {
  key: Exclude<DateKey, "any">;
  id: string;
  kicker: string;
  caption: string;
  bg: "bone" | "bone-deep";
  matchState: EventState;
  highlight?: boolean;
}[] = [
  {
    key: "upcoming",
    id: "upcoming",
    kicker: "Upcoming",
    caption: "Save the date · Photos coming soon",
    bg: "bone-deep",
    matchState: "upcoming",
  },
  {
    key: "live",
    id: "live",
    kicker: "Live now",
    caption: "Race day · Photos uploading",
    bg: "bone",
    matchState: "live",
    highlight: true,
  },
  {
    key: "recent",
    id: "recent",
    kicker: "Recent",
    caption: "Last 30 days · Photos all in",
    bg: "bone-deep",
    matchState: "open",
  },
  {
    key: "archive",
    id: "archive",
    kicker: "Archive",
    caption: "Older races · Still searchable",
    bg: "bone",
    matchState: "past",
  },
];

export function EventsBrowser({ events: seed }: EventsBrowserProps) {
  // Apply admin overrides on top of the SSR-passed seed so admin state
  // changes propagate to /events without re-fetching.
  const events = useEventCatalog(seed);
  const [query, setQuery] = useUrlState<string>("q", "", { debounceMs: 300 });
  const [city, setCity] = useUrlState<string>("city", "all");
  const [date, setDate] = useUrlState<DateKey>("date", "any");
  const setUrlBatch = useUrlStateBatch();

  const cityOptions = useMemo(() => {
    const unique = Array.from(new Set(events.map((e) => e.city)));
    unique.sort();
    return unique;
  }, [events]);

  const trimmed = query.trim().toLowerCase();
  const isFiltered = trimmed !== "" || city !== "all" || date !== "any";

  const filtered = useMemo(() => {
    return events.filter((e) => {
      if (city !== "all" && e.city !== city) return false;
      if (date !== "any") {
        const want = DATE_OPTIONS.find((d) => d.value === date)?.matchState;
        if (want && e.state !== want) return false;
      }
      if (trimmed) {
        const hay =
          `${e.name} ${e.location} ${e.city} ${e.date}`.toLowerCase();
        if (!hay.includes(trimmed)) return false;
      }
      return true;
    });
  }, [events, city, date, trimmed]);

  const liveCount = events.filter((e) => e.state === "live").length;

  const handleClearFilters = () => {
    setUrlBatch({ city: null, date: null });
  };

  return (
    <>
      <Hero liveCount={liveCount} />
      <FilterStrip
        query={query}
        onQueryChange={setQuery}
        city={city}
        onCityChange={setCity}
        date={date}
        onDateChange={setDate}
        cityOptions={cityOptions}
        onClearFilters={handleClearFilters}
      />
      <Results
        events={filtered}
        totalCount={events.length}
        isFiltered={isFiltered}
        city={city}
        date={date}
        filterKey={`${trimmed}|${city}|${date}`}
      />
    </>
  );
}

function Hero({ liveCount }: { liveCount: number }) {
  return (
    <section className="relative px-6 md:px-10 pt-14 md:pt-20 pb-12 md:pb-16 bg-bone overflow-hidden">
      <HeroVisual />

      <div className="relative max-w-7xl mx-auto">
        <div className="max-w-2xl stagger-children">
          <div className="flex items-center gap-3 mb-4">
            <span className="race-stripe" aria-hidden="true">
              <span className="bg-fresh" />
              <span className="bg-fresh-deep" />
              <span className="bg-ink" />
            </span>
            <Kicker as="p" size="md">
              Race photos · Cebu
            </Kicker>
          </div>
          <h1 className="font-hero text-ink text-5xl md:text-7xl lg:text-8xl">
            Pick your race.
            <br />
            <span className="text-fresh">Find your photos.</span>
          </h1>
          <p className="font-sans text-lg md:text-xl text-ink-soft max-w-md mt-6">
            Open an event, then search by face or bib. Free to look — pay only
            for the ones you keep.
          </p>

          {liveCount > 0 && (
            <div className="mt-10">
              <LiveTodayBlock liveCount={liveCount} />
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function HeroVisual() {
  return (
    <div
      aria-hidden="true"
      className="hidden lg:block absolute right-0 top-8 w-[48%] max-w-[600px] h-[460px] pointer-events-none"
    >
      <div
        className="absolute top-0 right-24 w-36 h-24 rounded-2xl bg-ink/[0.07]"
        style={{ animation: "fade-up 0.6s 0.6s both", opacity: 0 }}
      />
      <div
        className="absolute top-6 right-0 w-44 h-32 rounded-2xl bg-ink/[0.10]"
        style={{ animation: "fade-up 0.6s 0.75s both", opacity: 0 }}
      />
      <div
        className="absolute top-36 left-0 w-28 h-36 rounded-2xl bg-ink/[0.08]"
        style={{ animation: "fade-up 0.6s 0.9s both", opacity: 0 }}
      />
      <div
        className="absolute top-60 right-8 w-40 h-28 rounded-2xl bg-ink/[0.11]"
        style={{ animation: "fade-up 0.6s 1.05s both", opacity: 0 }}
      />
      <div
        className="absolute top-72 left-44 w-32 h-24 rounded-2xl bg-ink/[0.07]"
        style={{ animation: "fade-up 0.6s 1.2s both", opacity: 0 }}
      />

      <div
        className="absolute top-20 left-20 w-56 h-40 rounded-2xl bg-bone border-2 border-fresh shadow-[0_24px_40px_-12px_rgba(17,17,17,0.22)] overflow-hidden"
        style={{ animation: "fade-up 0.6s 1.35s both", opacity: 0 }}
      >
        <span className="absolute top-3 left-4 font-mono uppercase tracking-[0.14em] text-[9px] text-fresh">
          Match · 98%
        </span>
        <span className="absolute bottom-3 right-3 size-2.5 rounded-full bg-fresh breathe" />
        <svg
          className="absolute inset-0 m-auto size-10 text-ink/15"
          viewBox="0 0 32 32"
          fill="none"
          aria-hidden="true"
        >
          <rect
            x="4"
            y="7"
            width="24"
            height="18"
            rx="2"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <circle
            cx="16"
            cy="16"
            r="5"
            stroke="currentColor"
            strokeWidth="1.5"
          />
          <circle cx="22" cy="11" r="0.8" fill="currentColor" />
        </svg>
      </div>

      <div
        className="absolute top-[268px] left-[120px] font-mono uppercase tracking-[0.14em] text-[10px] text-slate"
        style={{ animation: "fade-in 0.6s 1.6s both", opacity: 0 }}
      >
        BIB · 4082
      </div>
      <div
        className="absolute top-[88px] right-[16px] font-mono uppercase tracking-[0.14em] text-[10px] text-slate"
        style={{ animation: "fade-in 0.6s 1.75s both", opacity: 0 }}
      >
        00:34:21
      </div>
    </div>
  );
}

function LiveTodayBlock({ liveCount }: { liveCount: number }) {
  const noun = liveCount === 1 ? "race" : "races";
  return (
    <a
      href="#live"
      aria-label={`Jump to ${liveCount} live ${noun} today`}
      className="group block rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-4 focus-visible:ring-offset-bone"
    >
      <Kicker as="p" tone="active" size="md" className="flex items-center gap-2">
        <span className="size-1.5 rounded-full bg-fresh breathe" />
        Live today
      </Kicker>
      <p className="font-display text-6xl md:text-7xl font-extrabold tracking-tight leading-none text-ink mt-2 tnum count-up">
        {liveCount}
      </p>
      <Kicker as="p" className="mt-2 group-hover:text-fresh transition-colors">
        {noun} · jump ↓
      </Kicker>
    </a>
  );
}

function FilterStrip({
  query,
  onQueryChange,
  city,
  onCityChange,
  date,
  onDateChange,
  cityOptions,
  onClearFilters,
}: {
  query: string;
  onQueryChange: (v: string) => void;
  city: string;
  onCityChange: (v: string) => void;
  date: DateKey;
  onDateChange: (v: DateKey) => void;
  cityOptions: string[];
  onClearFilters: () => void;
}) {
  const cityLabel = city === "all" ? "All cities" : city;
  const dateLabel =
    DATE_OPTIONS.find((d) => d.value === date)?.label ?? "Any time";
  const hasActiveFilter = city !== "all" || date !== "any";
  const handleClearFilters = onClearFilters;

  return (
    <div className="sticky top-[var(--site-header-h)] z-20 bg-bone/90 backdrop-blur-md border-y border-line">
      <div className="max-w-7xl mx-auto px-6 md:px-10 py-3">
        <div className="flex flex-col md:flex-row md:items-center gap-3 md:gap-8">
          <div className="md:flex-1 min-w-0">
            <StripSearchInput value={query} onChange={onQueryChange} />
          </div>
          <div className="flex items-center justify-end gap-5 md:gap-8 md:shrink-0">
            <Dropdown
              ariaLabel="Filter by city"
              align="right"
              trigger={<DropdownTrigger>{cityLabel}</DropdownTrigger>}
            >
              <DropdownItem
                variant="mono"
                active={city === "all"}
                onClick={() => onCityChange("all")}
              >
                All cities
              </DropdownItem>
              {cityOptions.map((c) => (
                <DropdownItem
                  key={c}
                  variant="mono"
                  active={c === city}
                  onClick={() => onCityChange(c)}
                >
                  {c}
                </DropdownItem>
              ))}
            </Dropdown>
            <Dropdown
              ariaLabel="Filter by date"
              align="right"
              trigger={<DropdownTrigger>{dateLabel}</DropdownTrigger>}
            >
              {DATE_OPTIONS.map((d) => (
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
            {hasActiveFilter && (
              <>
                <span
                  aria-hidden="true"
                  className="h-3 w-px bg-line shrink-0"
                />
                <Kicker
                  as="button"
                  type="button"
                  onClick={handleClearFilters}
                  aria-label="Clear filters"
                  className="inline-flex items-center gap-1.5 rounded-sm hover:text-fresh transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone shrink-0"
                >
                  Clear
                  <span aria-hidden="true">✕</span>
                </Kicker>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StripSearchInput({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
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
        placeholder="Search by event name…"
        aria-label="Search events"
        className="block w-full pl-7 pr-2 py-1.5 bg-transparent border-b border-line focus:border-fresh transition-colors outline-none font-mono text-sm tracking-wide placeholder:text-slate-soft text-ink"
      />
    </div>
  );
}

function Results({
  events,
  totalCount,
  isFiltered,
  city,
  date,
  filterKey,
}: {
  events: ReadonlyArray<ListEvent>;
  totalCount: number;
  isFiltered: boolean;
  city: string;
  date: DateKey;
  filterKey: string;
}) {
  if (events.length === 0) {
    return <EmptyState />;
  }

  if (!isFiltered) {
    return (
      <>
        {SEGMENT_GROUPS.map((g) => {
          const items = events.filter((e) => e.state === g.matchState);
          if (items.length === 0) return null;
          return (
            <SegmentBlock
              key={g.key}
              id={g.id}
              kicker={g.kicker}
              caption={g.caption}
              bg={g.bg}
              items={items}
              highlight={g.highlight}
            />
          );
        })}
      </>
    );
  }

  return (
    <FlatResults
      events={events}
      totalCount={totalCount}
      city={city}
      date={date}
      filterKey={filterKey}
    />
  );
}

function SegmentBlock({
  id,
  kicker,
  caption,
  bg,
  items,
  highlight = false,
}: {
  id: string;
  kicker: string;
  caption: string;
  bg: "bone" | "bone-deep";
  items: ReadonlyArray<ListEvent>;
  highlight?: boolean;
}) {
  const bgClass = bg === "bone-deep" ? "bg-bone-deep" : "bg-bone";
  return (
    <section id={id} className={`${bgClass} px-6 md:px-10 py-14 md:py-20 scroll-mt-28`}>
      <div className="max-w-7xl mx-auto">
        <div className="flex items-baseline justify-between gap-6 mb-10 md:mb-12 border-b border-line pb-4">
          <div className="flex items-baseline gap-3 min-w-0">
            {highlight && (
              <span className="size-2 rounded-full bg-fresh breathe self-center shrink-0" />
            )}
            <Kicker
              as="p"
              tone={highlight ? "active" : "default"}
              size="md"
              className="shrink-0"
            >
              {kicker}
            </Kicker>
            <Kicker as="p" tone="soft" className="hidden sm:block truncate">
              {caption}
            </Kicker>
          </div>
          <Kicker as="p" tone="soft" tnum className="shrink-0">
            {items.length} race{items.length === 1 ? "" : "s"}
          </Kicker>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {items.map((e, i) => (
            <EventTile key={e.id} event={e} index={i} />
          ))}
        </div>
      </div>
    </section>
  );
}

function FlatResults({
  events,
  totalCount,
  city,
  date,
  filterKey,
}: {
  events: ReadonlyArray<ListEvent>;
  totalCount: number;
  city: string;
  date: DateKey;
  filterKey: string;
}) {
  const cityLabel = city === "all" ? "All cities" : city;
  const dateLabel =
    DATE_OPTIONS.find((d) => d.value === date)?.label ?? "Any time";
  const noun = events.length === 1 ? "race" : "races";
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.EVENT_GRID_INITIAL);

  useEffect(() => {
    setLoadedCount(PAGE_SIZE.EVENT_GRID_INITIAL);
  }, [filterKey]);

  const visibleSlice = events.slice(0, loadedCount);

  return (
    <section className="bg-bone px-6 md:px-10 py-14 md:py-20">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-baseline justify-between gap-6 mb-10 md:mb-12 border-b border-line pb-4">
          <Kicker as="p" size="md">
            <span className="tnum text-ink">{events.length}</span>{" "}
            <span className="text-slate-soft">of</span>{" "}
            <span className="tnum">{totalCount}</span> {noun}
          </Kicker>
          <Kicker as="p" tone="soft" className="truncate">
            {cityLabel} · {dateLabel}
          </Kicker>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {visibleSlice.map((e, i) => (
            <EventTile key={e.id} event={e} index={i} />
          ))}
        </div>
        <LoadMoreButton
          shown={visibleSlice.length}
          total={events.length}
          increment={PAGE_SIZE.EVENT_GRID_INCREMENT}
          onLoadMore={() =>
            setLoadedCount((n) => n + PAGE_SIZE.EVENT_GRID_INCREMENT)
          }
        />
      </div>
    </section>
  );
}

function EmptyState() {
  return (
    <section className="px-6 md:px-10 py-24 md:py-32 bg-bone min-h-[40vh] flex items-center">
      <div className="max-w-2xl mx-auto w-full text-center">
        <Kicker as="p" className="mb-3">
          No matches
        </Kicker>
        <p className="font-display text-3xl md:text-4xl font-extrabold text-ink tracking-tight">
          Nothing here yet.
        </p>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-4">
          Try clearing the search, or change the city or date above.
        </p>
      </div>
    </section>
  );
}

