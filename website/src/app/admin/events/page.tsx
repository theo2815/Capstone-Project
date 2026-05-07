"use client";

import { useMemo } from "react";
import { Slab } from "@/components/profile-shell";
import { AdminEventCard } from "@/components/admin/admin-event-card";
import { useEventCatalog } from "@/lib/event-catalog";
import type { EventState, ListEvent } from "@/app/events/events-browser";

// Phase 1 admin Events page. Four slabs (Live / Upcoming / Recent /
// Archive) of <AdminEventCard>s with inline state-change dropdowns.
// Reads `useEventCatalog()` so admin overrides persisted in
// `admin-event-overrides-store` flow through. The same hook backs
// /events and /dashboard/upload, closing the demo loop end-to-end.

const SECTIONS: ReadonlyArray<{
  id: string;
  number: string;
  title: string;
  caption: string;
  matchState: EventState;
  cols: string;
}> = [
  {
    id: "live",
    number: "01",
    title: "Live now",
    caption: "Currently accepting uploads",
    matchState: "live",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "upcoming",
    number: "02",
    title: "Upcoming",
    caption: "Race day pending",
    matchState: "upcoming",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "recent",
    number: "03",
    title: "Recent",
    caption: "Galleries open for sale",
    matchState: "open",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "archive",
    number: "04",
    title: "Archive",
    caption: "Read-only history",
    matchState: "past",
    cols: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
  },
];

export default function AdminEventsPage() {
  const catalog = useEventCatalog();

  const grouped = useMemo(() => {
    return SECTIONS.map((section) => ({
      ...section,
      items: byState(catalog, section.matchState),
    }));
  }, [catalog]);

  const liveCount = byState(catalog, "live").length;

  return (
    <>
      <Header total={catalog.length} liveCount={liveCount} />
      {grouped.map((section) => {
        const noun = section.items.length === 1 ? "event" : "events";
        return (
          <Slab
            key={section.id}
            id={section.id}
            number={section.number}
            title={section.title}
            caption={section.caption}
            trailing={`${section.items.length} ${noun}`}
          >
            {section.items.length === 0 ? (
              <p className="font-sans text-sm text-slate-soft">
                {emptyCopyFor(section.matchState)}
              </p>
            ) : (
              <div className={`grid gap-6 md:gap-8 ${section.cols}`}>
                {section.items.map((event, i) => (
                  <AdminEventCard key={event.id} event={event} index={i} />
                ))}
              </div>
            )}
          </Slab>
        );
      })}
    </>
  );
}

function Header({
  total,
  liveCount,
}: {
  total: number;
  liveCount: number;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Events ·{" "}
        <span className="tnum">{total.toLocaleString()}</span> on platform ·{" "}
        <span className="tnum">{liveCount}</span> live
      </p>
      <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
        Events.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Move a race through its lifecycle. Flipping a state here updates
        the runner-facing /events listing and the photographer&apos;s upload
        picker the moment you save.
      </p>
    </header>
  );
}

function byState(
  catalog: ReadonlyArray<ListEvent>,
  state: EventState,
): ListEvent[] {
  const matching = catalog.filter((e) => e.state === state);
  return matching.sort((a, b) =>
    state === "upcoming"
      ? a.date.localeCompare(b.date)
      : b.date.localeCompare(a.date),
  );
}

function emptyCopyFor(state: EventState): string {
  switch (state) {
    case "live":
      return "No live events right now.";
    case "upcoming":
      return "No upcoming races on the calendar.";
    case "open":
      return "No recent galleries open for sale.";
    case "past":
      return "No archived events yet.";
  }
}
