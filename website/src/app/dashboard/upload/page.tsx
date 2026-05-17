"use client";

import { useMemo, useState } from "react";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { Slab } from "@/components/profile-shell";
import {
  EventFilterBar,
  EventFilterEmpty,
  matchEventDate,
  type EventDateKey,
} from "@/components/dashboard/event-filter-bar";
import { EventTile } from "@/components/events/event-tile";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { useCanUpload } from "@/hooks/use-can-upload";
import { usePublicEvents } from "@/hooks/use-public-events";
import { useEventCatalog } from "@/lib/event-catalog";
import { PAGE_SIZE } from "@/lib/pagination-config";

// Stable empty-array reference so useEventCatalog's memo doesn't churn while
// the public events list is loading.
const EMPTY_SEED: ReadonlyArray<ListEvent> = [];

// Photographer upload picker. Shows every event in the catalog (not just the
// photographer's own coverage) so they can pick where to lift photos to. The
// per-photographer view of "events I covered" lives at /dashboard/events.

const SECTIONS: Array<{
  id: string;
  number: string;
  title: string;
  caption: string;
  matchState: EventState;
}> = [
  {
    id: "live",
    number: "01",
    title: "Live now",
    caption: "Race day · upload as you shoot",
    matchState: "live",
  },
  {
    id: "upcoming",
    number: "02",
    title: "Upcoming",
    caption: "Pre-stage and test your kit",
    matchState: "upcoming",
  },
  {
    id: "recent",
    number: "03",
    title: "Recent",
    caption: "Last 30 days · late frames welcome",
    matchState: "open",
  },
  {
    id: "archive",
    number: "04",
    title: "Archive",
    caption: "Older races · keep adding if you have them",
    matchState: "past",
  },
];

export default function UploadPickerPage() {
  // VerificationBanner + verification sync live in DashboardShell so the
  // suspended/incomplete/pending banner stays consistent across every
  // /dashboard/* page without each page re-mounting them.
  const gate = useCanUpload();
  const canUpload = gate.kind === "ok";
  const liveEvents = usePublicEvents();
  const catalog = useEventCatalog(liveEvents ?? EMPTY_SEED);

  const [date, setDate] = useState<EventDateKey>("any");
  const [query, setQuery] = useState("");

  const filteredCatalog = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    return catalog.filter((event) => {
      if (!matchEventDate(event.state, date)) return false;
      if (trimmed) {
        const hay = `${event.name} ${event.location} ${event.city}`.toLowerCase();
        if (!hay.includes(trimmed)) return false;
      }
      return true;
    });
  }, [catalog, date, query]);

  const visibleSections = SECTIONS.map((section) => ({
    ...section,
    items: filteredCatalog.filter((e) => e.state === section.matchState),
  })).filter((section) => section.items.length > 0);

  const isFiltered = date !== "any" || query.trim().length > 0;

  const clearFilters = () => {
    setDate("any");
    setQuery("");
  };

  // Catalog itself is empty — no events to upload to at all. Skip the filter
  // bar (filtering nothing yields nothing) and show the wider PickerEmpty.
  if (catalog.length === 0) {
    return <PickerEmpty />;
  }

  return (
    <>
      <EventFilterBar
        date={date}
        onDateChange={setDate}
        query={query}
        onQueryChange={setQuery}
        dateAriaLabel="Filter upload picker by date"
        searchAriaLabel="Search events to upload to"
      />

      {visibleSections.length === 0 ? (
        isFiltered ? (
          <EventFilterEmpty onClear={clearFilters} />
        ) : (
          <PickerEmpty />
        )
      ) : (
        visibleSections.map((section) => (
          <UploadSection
            key={section.id}
            id={section.id}
            number={section.number}
            title={section.title}
            caption={section.caption}
            items={section.items}
            canUpload={canUpload}
          />
        ))
      )}
    </>
  );
}

function UploadSection({
  id,
  number,
  title,
  caption,
  items,
  canUpload,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  items: ReadonlyArray<ListEvent>;
  canUpload: boolean;
}) {
  const noun = items.length === 1 ? "race" : "races";
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.EVENT_GRID_INITIAL);
  const visibleSlice = items.slice(0, loadedCount);

  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${items.length} ${noun}`}
    >
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
        {visibleSlice.map((event, i) => (
          <EventTile
            key={event.id}
            mode="upload"
            event={event}
            index={i}
            canUpload={canUpload}
          />
        ))}
      </div>
      <LoadMoreButton
        shown={visibleSlice.length}
        total={items.length}
        increment={PAGE_SIZE.EVENT_GRID_INCREMENT}
        onLoadMore={() =>
          setLoadedCount((n) => n + PAGE_SIZE.EVENT_GRID_INCREMENT)
        }
      />
    </Slab>
  );
}

function PickerEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No events to upload to.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Schedule a coverage to start lifting photos.
      </p>
    </div>
  );
}
