"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { EventCouponModal } from "@/components/dashboard/event-coupon-modal";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import {
  EventFilterBar,
  EventFilterEmpty,
  matchEventDate,
  type EventDateKey,
} from "@/components/dashboard/event-filter-bar";
import { DashboardEventRow } from "@/components/dashboard/dashboard-event-row";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
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
import type { ListEvent } from "@/types/event";
import type { PhotographerEventSummary } from "@/lib/photographer-mock";
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
        .map((p) => ({ catalog: summaryToListEvent(p), photographer: p })),
    [liveEvents],
  );

  const [date, setDate] = useState<EventDateKey>("any");
  const [query, setQuery] = useState("");
  const [couponOpen, setCouponOpen] = useState(false);
  const couponEvents = useMemo(
    () => covered.map(({ photographer }) => photographer),
    [covered],
  );

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

  // Two clear buckets: events the photographer created (ownedByMe) vs admin
  // races they're covering. The BE already stamps ownedByMe per row.
  const mine = useMemo(
    () => filtered.filter(({ photographer }) => photographer.ownedByMe),
    [filtered],
  );
  const adminCovered = useMemo(
    () => filtered.filter(({ photographer }) => !photographer.ownedByMe),
    [filtered],
  );

  const clearFilters = () => {
    setDate("any");
    setQuery("");
  };

  if (isLoading) {
    return (
      <Slab
        id="mine"
        number="01"
        title="My events"
        caption="Races you've created or covered"
      >
        <div className="space-y-2">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20 w-full rounded-xl" />
          ))}
        </div>
      </Slab>
    );
  }

  return (
    <>
      <div className="flex flex-wrap justify-end gap-2 pt-10 md:pt-20">
        {/* Mobile has no bottom tab for removal requests; this and the inbox link reach them. */}
        <Link href={ROUTES.DASHBOARD_PHOTO_REMOVAL_REQUESTS} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
          Removal requests
        </Link>
        <button
          type="button"
          onClick={() => setCouponOpen(true)}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          + Create event coupon
        </button>
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
            dateAriaLabel="Filter your events by date"
            searchAriaLabel="Search your events"
          />
          {mine.length === 0 && adminCovered.length === 0 ? (
            <EventFilterEmpty onClear={clearFilters} />
          ) : (
            <>
              {mine.length > 0 && (
                <EventGroupSlab
                  id="mine"
                  number="01"
                  title="My events"
                  caption="Events you created"
                  items={mine}
                />
              )}
              {adminCovered.length > 0 && (
                <EventGroupSlab
                  id="admin"
                  number={mine.length > 0 ? "02" : "01"}
                  title="Race events"
                  caption="Shared races you're covering or submitted"
                  items={adminCovered}
                />
              )}
            </>
          )}
        </>
      )}
      <EventCouponModal
        isOpen={couponOpen}
        onClose={() => setCouponOpen(false)}
        events={couponEvents}
      />
    </>
  );
}

// One "My events" / "Race events" section: a Slab with its own paginated grid.
// Each group owns its load-more so the two lists page independently.
function EventGroupSlab({
  id,
  number,
  title,
  caption,
  items,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  items: Array<{ catalog: ListEvent; photographer: PhotographerEventSummary }>;
}) {
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.EVENT_GRID_INITIAL);
  const visibleSlice = items.slice(0, loadedCount);
  const noun = items.length === 1 ? "event" : "events";

  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${items.length} ${noun}`}
    >
      <ul className="border border-line rounded-2xl bg-surface divide-y divide-line overflow-hidden">
        {visibleSlice.map(({ catalog, photographer }) => (
          <li key={catalog.id}>
            <DashboardEventRow
              event={catalog}
              photoCount={photographer.photoCount}
              salesCount={photographer.salesCount}
              note={ownedEventNote(photographer)}
            />
          </li>
        ))}
      </ul>
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
      <div className="mt-6">
        <CreateEventLink />
      </div>
    </div>
  );
}
