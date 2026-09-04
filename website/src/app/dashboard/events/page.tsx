"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import {
  EventFilterBar,
  EventFilterEmpty,
  matchEventDate,
  type EventDateKey,
} from "@/components/dashboard/event-filter-bar";
import { EventTile } from "@/components/events/event-tile";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { TileSkeleton } from "@/components/ui/skeleton";
import {
  usePhotographerEvents,
  COVERED_EVENTS_MAX,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  ownedEventNote,
  summaryToListEvent,
} from "@/lib/photographer-events";
import { cn } from "@/lib/utils";

// Photographer's events list — races they've uploaded to, plus every event
// they created themselves (V46), which shows up from the moment it is
// submitted so its review chip is visible. Each card routes to the focused
// share page at /dashboard/events/[id].

export default function DashboardEventsPage() {
  // withUploads:false — a just-submitted owned event has no photos yet and
  // the server-side filter would hide it. Coverage rows for admin events
  // only exist after a first upload, so nothing else sneaks in.
  const liveEvents = usePhotographerEvents({
    withUploads: false,
    limit: COVERED_EVENTS_MAX,
  });
  const isLoading = liveEvents === null;

  const covered = useMemo(
    () =>
      (liveEvents ?? [])
        .filter((p) => p.photoCount > 0 || p.ownedByMe)
        .map((p) => ({ catalog: summaryToListEvent(p), photographer: p })),
    [liveEvents],
  );

  const trailing =
    covered.length > 0
      ? `${covered.length} event${covered.length === 1 ? "" : "s"}`
      : undefined;

  const [date, setDate] = useState<EventDateKey>("any");
  const [query, setQuery] = useState("");
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.EVENT_GRID_INITIAL);

  const filtered = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    return covered.filter(({ catalog }) => {
      if (!matchEventDate(catalog.state, date)) return false;
      if (trimmed) {
        const hay =
          `${catalog.name} ${catalog.location} ${catalog.city}`.toLowerCase();
        if (!hay.includes(trimmed)) return false;
      }
      return true;
    });
  }, [covered, date, query]);

  const visibleSlice = filtered.slice(0, loadedCount);
  const clearFilters = () => {
    setDate("any");
    setQuery("");
  };

  if (isLoading) {
    return (
      <Slab
        id="covered"
        number="01"
        title="Your events"
        caption="Races you've covered or created"
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
          {[0, 1, 2, 3].map((i) => (
            <TileSkeleton key={i} />
          ))}
        </div>
      </Slab>
    );
  }

  return (
    <Slab
      id="covered"
      number="01"
      title="Your events"
      caption="Races you've covered or created"
      trailing={trailing}
    >
      <div className="flex justify-end mb-6">
        <CreateEventLink />
      </div>
      {covered.length === 0 ? (
        <CoveredEmpty />
      ) : (
        <>
          <EventFilterBar
            date={date}
            onDateChange={setDate}
            query={query}
            onQueryChange={setQuery}
            dateAriaLabel="Filter covered events by date"
            searchAriaLabel="Search covered events"
          />
          {filtered.length === 0 ? (
            <EventFilterEmpty onClear={clearFilters} />
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
                {visibleSlice.map(({ catalog, photographer }, i) => (
                  <EventTile
                    key={catalog.id}
                    mode="manage"
                    event={catalog}
                    index={i}
                    photoCount={photographer.photoCount}
                    salesCount={photographer.salesCount}
                    note={ownedEventNote(photographer)}
                  />
                ))}
              </div>
              <LoadMoreButton
                shown={visibleSlice.length}
                total={filtered.length}
                increment={PAGE_SIZE.EVENT_GRID_INCREMENT}
                onLoadMore={() =>
                  setLoadedCount((n) => n + PAGE_SIZE.EVENT_GRID_INCREMENT)
                }
              />
            </>
          )}
        </>
      )}
    </Slab>
  );
}

function CreateEventLink() {
  return (
    <Link
      href={`${ROUTES.DASHBOARD_EVENTS}/new`}
      className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
    >
      + Create an event
    </Link>
  );
}

function CoveredEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No events yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Upload your first batch to a race from the Upload page, or create an
        event of your own — paid or free, public or link-only.
      </p>
    </div>
  );
}
