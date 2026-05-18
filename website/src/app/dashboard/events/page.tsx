"use client";

import { useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import {
  EventFilterBar,
  EventFilterEmpty,
  matchEventDate,
  type EventDateKey,
} from "@/components/dashboard/event-filter-bar";
import { EventTile } from "@/components/events/event-tile";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { TileSkeleton } from "@/components/ui/skeleton";
import { usePhotographerEvents } from "@/hooks/use-photographer-data";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { PAGE_SIZE } from "@/lib/pagination-config";
import type { PhotographerEventSummary } from "@/lib/photographer-mock";

// Photographer's "events I've covered" list — only events with at least one
// uploaded photo show up. Each card routes to the focused share page at
// /dashboard/events/[id].

// Build a ListEvent-shaped record from the BE PhotographerEventSummary so
// EventTile (which expects ListEvent) can render it. The BE response already
// carries every field manage-mode uses (id/slug/name/date/location/state);
// participantCount + status are slot fillers for manage mode and aren't read.
function toListEvent(p: PhotographerEventSummary): ListEvent {
  return {
    id: p.id,
    slug: p.slug,
    name: p.name,
    date: p.date,
    location: p.location,
    bannerUrl: p.bannerUrl ?? undefined,
    photoCount: p.photoCount,
    participantCount: 0,
    status: "ACTIVE",
    state: p.state as EventState,
    // "City Center, Cebu City" → "Cebu City". Manage mode doesn't currently
    // surface city, but ListEvent requires the field — derive it cheaply
    // instead of querying a second BE endpoint.
    city: p.location.split(",").pop()?.trim() ?? "",
  };
}

export default function DashboardEventsPage() {
  const liveEvents = usePhotographerEvents({ withUploads: true });
  const isLoading = liveEvents === null;

  const covered = useMemo(
    () =>
      (liveEvents ?? [])
        .filter((p) => p.photoCount > 0)
        .map((p) => ({ catalog: toListEvent(p), photographer: p })),
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
        title="Your covered events"
        caption="Events where you've uploaded photos"
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
      title="Your covered events"
      caption="Events where you've uploaded photos"
      trailing={trailing}
    >
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

function CoveredEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No events covered yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Once you upload your first batch from the Upload page, the event lands
        here so you can share its public gallery.
      </p>
    </div>
  );
}
