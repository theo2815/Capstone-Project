"use client";

import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Slab } from "@/components/profile-shell";
import { AdminEventCard } from "@/components/admin/admin-event-card";
import { AdminEventFormModal } from "@/components/admin/admin-event-form-modal";
import { useEventCatalog } from "@/lib/event-catalog";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";
import { useAdminEvents } from "@/hooks/use-admin-data";
import { useToast } from "@/hooks/use-toast";
import { useConfirmation } from "@/hooks/use-confirmation";
import type { EventState, ListEvent } from "@/app/events/events-browser";

// Admin Events page. Four slabs (Live / Upcoming / Recent / Archive) of
// <AdminEventCard>s. Lifecycle state is derived from `event.date`
// (see `lib/event-catalog.ts`) — admin only edits Title/Location/Date and
// the four states fall out automatically:
//
//   future date            → Upcoming
//   day 0..3 (race + 4d)   → Live  (photographer upload window)
//   day 4..89              → Recent (gallery for sale, no more uploads)
//   day 90+                → Archive
//
// The same `useEventCatalog()` backs /events and /dashboard/upload, so a
// create/edit/delete here flows into both runner + photographer surfaces.

// Stable empty-array reference so `useEventCatalog`'s memo doesn't churn
// while the live admin list is loading. Passing a fresh `[]` every render
// would rebuild the catalog on every paint.
const EMPTY_SEED: ReadonlyArray<ListEvent> = [];

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
    caption: "Currently accepting uploads · 4-day grace from race day",
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
    caption: "Galleries open for sale · upload window closed",
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
  // Live admin list — the source of truth across refreshes. The local
  // override store layers optimistic submissions / edits / deletes on top
  // until the next refetch picks them up from the backend.
  const liveEvents = useAdminEvents();
  const catalog = useEventCatalog(liveEvents ?? EMPTY_SEED);
  const createEvent = useAdminEventOverridesStore((s) => s.createEvent);
  const editEvent = useAdminEventOverridesStore((s) => s.editEvent);
  const deleteEvent = useAdminEventOverridesStore((s) => s.deleteEvent);
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const { confirm } = useConfirmation();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<ListEvent | null>(null);

  const grouped = useMemo(() => {
    return SECTIONS.map((section) => ({
      ...section,
      items: byState(catalog, section.matchState),
    }));
  }, [catalog]);

  const liveCount = byState(catalog, "live").length;

  async function handleCreateSubmit(payload: {
    name: string;
    location: string;
    date: string;
    cover: File | null;
    removeCover: boolean;
  }) {
    try {
      await createEvent({
        name: payload.name,
        location: payload.location,
        date: payload.date,
        cover: payload.cover,
      });
      // Refetch the admin list so the optimistic submission gets replaced
      // by the canonical backend row on the next paint — and so the row
      // survives a manual refresh.
      void queryClient.invalidateQueries({ queryKey: ["admin", "events"] });
      setCreateOpen(false);
      showToast({ kind: "success", message: `Created · ${payload.name}` });
    } catch (err) {
      // Keep the modal open so the admin can retry without losing the
      // already-typed fields. ApiError carries the backend's user-facing
      // message; everything else falls back to a generic copy.
      const message =
        err instanceof Error && err.message ? err.message : "Could not create event.";
      showToast({ kind: "error", message });
    }
  }

  async function handleEditSubmit(payload: {
    name: string;
    location: string;
    date: string;
    cover: File | null;
    removeCover: boolean;
  }) {
    if (!editTarget) return;
    try {
      await editEvent(editTarget.id, {
        name: payload.name,
        location: payload.location,
        date: payload.date,
        cover: payload.cover,
        removeCover: payload.removeCover,
      });
      // Refetch so the new presigned cover URL (or the cleared cover) lands
      // on the next paint — the local override only carries field changes.
      void queryClient.invalidateQueries({ queryKey: ["admin", "events"] });
      setEditTarget(null);
      showToast({ kind: "success", message: `Updated · ${payload.name}` });
    } catch (err) {
      const message =
        err instanceof Error && err.message ? err.message : "Could not update event.";
      showToast({ kind: "error", message });
    }
  }

  async function handleDelete(event: ListEvent) {
    const ok = await confirm({
      title: "Delete event?",
      message: (
        <>
          Removes <strong className="text-ink">{event.name}</strong> from{" "}
          <span className="font-mono text-[13px]">/events</span>, the
          photographer upload picker, and the admin board. This is a mock
          delete — the real backend will soft-archive instead.
        </>
      ),
      confirmLabel: "Delete event",
      danger: true,
    });
    if (!ok) return;
    deleteEvent(event.id);
    showToast({ kind: "info", message: `Deleted · ${event.name}` });
  }

  return (
    <>
      <Header
        total={catalog.length}
        liveCount={liveCount}
        onCreate={() => setCreateOpen(true)}
      />
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
                  <AdminEventCard
                    key={event.id}
                    event={event}
                    index={i}
                    onEdit={(e) => setEditTarget(e)}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            )}
          </Slab>
        );
      })}

      <AdminEventFormModal
        open={createOpen}
        mode="create"
        onClose={() => setCreateOpen(false)}
        onSubmit={handleCreateSubmit}
      />
      <AdminEventFormModal
        open={editTarget !== null}
        mode="edit"
        event={editTarget}
        onClose={() => setEditTarget(null)}
        onSubmit={handleEditSubmit}
      />
    </>
  );
}

function Header({
  total,
  liveCount,
  onCreate,
}: {
  total: number;
  liveCount: number;
  onCreate: () => void;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            Events ·{" "}
            <span className="tnum">{total.toLocaleString()}</span> on platform ·{" "}
            <span className="tnum">{liveCount}</span> live
          </p>
          <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
            Events.
          </h1>
        </div>
        <button
          type="button"
          onClick={onCreate}
          className="shrink-0 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] bg-fresh text-bone hover:bg-fresh-deep transition-colors rounded-full px-5 py-2.5"
        >
          + New event
        </button>
      </div>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Lifecycle is derived from the date — set Title, Location, and Date and
        the cards on /events plus the photographer upload picker update the
        moment you save. Photographers have a 4-day upload window starting on
        race day.
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
