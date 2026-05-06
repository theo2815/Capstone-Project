"use client";

import { Slab } from "@/components/profile-shell";
import { EventTile } from "@/components/events/event-tile";
import { getEventById } from "@/lib/event-catalog";
import { PHOTOGRAPHER_EVENTS } from "@/lib/photographer-mock";

// Photographer's "events I've covered" list — only events with at least one
// uploaded photo show up. Each card routes to the focused share page at
// /dashboard/events/[id].

export default function DashboardEventsPage() {
  const covered = PHOTOGRAPHER_EVENTS.filter((p) => p.photoCount > 0)
    .map((p) => {
      const catalog = getEventById(p.id);
      return catalog ? { catalog, photographer: p } : null;
    })
    .filter(
      (x): x is { catalog: NonNullable<ReturnType<typeof getEventById>>; photographer: (typeof PHOTOGRAPHER_EVENTS)[number] } =>
        x !== null,
    );

  const trailing =
    covered.length > 0
      ? `${covered.length} event${covered.length === 1 ? "" : "s"}`
      : undefined;

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
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
          {covered.map(({ catalog, photographer }, i) => (
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
