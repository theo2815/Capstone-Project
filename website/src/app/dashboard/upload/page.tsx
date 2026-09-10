"use client";

import { useMemo, useState } from "react";
import type { EventState, ListEvent } from "@/types/event";
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
import { usePhotographerEvents } from "@/hooks/use-photographer-data";
import { usePublicEvents } from "@/hooks/use-public-events";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  isOwnedEventLive,
  summaryToListEvent,
} from "@/lib/photographer-events";

// Stable empty catalog while
// the public events list is loading.
const EMPTY_SEED: ReadonlyArray<ListEvent> = [];

// Photographer upload picker. Shows every event in the catalog (not just the
// photographer's own coverage) so they can pick where to lift photos to. The
// per-photographer view of "events I covered" lives at /dashboard/events.

// Admin events surface uploadable-now first: live race day, then upcoming, then
// recently-open galleries, then the archive — date order preserved within each
// (the catalog arrives date-desc and the sort is stable).
const STATE_ORDER: Record<EventState, number> = {
  live: 0,
  upcoming: 1,
  open: 2,
  past: 3,
};

export default function UploadPickerPage() {
  // VerificationBanner + verification sync live in DashboardShell so the
  // suspended/incomplete/pending banner stays consistent across every
  // /dashboard/* page without each page re-mounting them.
  const gate = useCanUpload();
  const canUpload = gate.kind === "ok";
  const liveEvents = usePublicEvents();
  const catalog = liveEvents ?? EMPTY_SEED;
  // Photographer-owned events (V46) that are approved — the owner uploads
  // on any date, so they sit above the date-gated public sections. Unlisted
  // ones never reach the public catalog; this is their only way in.
  const mine = usePhotographerEvents();
  const owned = useMemo(
    () => (mine ?? []).filter(isOwnedEventLive).map(summaryToListEvent),
    [mine],
  );

  const [date, setDate] = useState<EventDateKey>("any");
  const [query, setQuery] = useState("");

  // Admin events only — photographer-owned events are excluded from the public
  // catalog server-side (EventRepository.search), so everything here is an
  // admin race a verified photographer may cover. "My events" comes from the
  // owner endpoint above.
  const adminEvents = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    return catalog
      .filter((event) => {
        if (!matchEventDate(event.state, date)) return false;
        if (trimmed) {
          const hay = `${event.name} ${event.location} ${event.city}`.toLowerCase();
          if (!hay.includes(trimmed)) return false;
        }
        return true;
      })
      .sort((a, b) => STATE_ORDER[a.state] - STATE_ORDER[b.state]);
  }, [catalog, date, query]);

  const isFiltered = date !== "any" || query.trim().length > 0;

  const clearFilters = () => {
    setDate("any");
    setQuery("");
  };

  // No events at all — skip the filter bar (filtering nothing yields nothing)
  // and show the wider PickerEmpty.
  if (catalog.length === 0 && owned.length === 0) {
    return <PickerEmpty />;
  }

  return (
    <>
      {owned.length > 0 && (
        <UploadSection
          id="mine"
          number="01"
          title="My events"
          caption="Events you created · upload up to a month after race day"
          items={owned}
          canUpload={canUpload}
          owned
        />
      )}

      {catalog.length > 0 && (
        <EventFilterBar
          date={date}
          onDateChange={setDate}
          query={query}
          onQueryChange={setQuery}
          dateAriaLabel="Filter admin events by date"
          searchAriaLabel="Search admin events to upload to"
        />
      )}

      {catalog.length > 0 &&
        (adminEvents.length === 0 ? (
          isFiltered ? (
            <EventFilterEmpty onClear={clearFilters} />
          ) : null
        ) : (
          <UploadSection
            id="admin"
            number={owned.length > 0 ? "02" : "01"}
            title="Admin events"
            caption="Races QuickPitik scheduled · race-day upload window"
            items={adminEvents}
            canUpload={canUpload}
          />
        ))}
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
  owned = false,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  items: ReadonlyArray<ListEvent>;
  canUpload: boolean;
  owned?: boolean;
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
            owned={owned}
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
