"use client";

import { useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import {
  AdminEventCard,
  type AdminEventReview,
} from "@/components/admin/admin-event-card";
import { AdminEventFormModal } from "@/components/admin/admin-event-form-modal";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";
import { useAdminEvents, useAdminEventActions, EMPTY_ADMIN_EVENTS } from "@/hooks/use-admin-events";
import { useToast } from "@/hooks/use-toast";
import { useConfirmation } from "@/hooks/use-confirmation";
import type { EventState, ListEvent } from "@/types/event";

// Admin Events page. Four slabs (Live / Upcoming / Recent / Archive) of
// <AdminEventCard>s. Lifecycle state is derived from `event.date`
// (see `lib/event-catalog.ts`) — admin only edits Title/Location/Date and
// the four states fall out automatically:
//
//   future date            → Upcoming
//   day 0..3 (race + 4d)   → Live  (photographer upload window)
//   day 4..89              → Recent (gallery for sale, no more uploads)
//   day 90+                → Archive

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
  const query = useAdminEvents();
  const catalog = query.data ?? EMPTY_ADMIN_EVENTS;
  const { createEvent, editEvent, deleteEvent, pendingAction, isPending, recoveryNotice } = useAdminEventActions();
  const { showToast } = useToast();
  const { confirm } = useConfirmation();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<ListEvent | null>(null);
  const [editOpen, setEditOpen] = useState(false);

  const grouped = useMemo(() => {
    return SECTIONS.map((section) => ({
      ...section,
      items: byState(catalog, section.matchState),
    }));
  }, [catalog]);

  // Owner and review metadata come from the same canonical admin rows.
  const reviewById = useMemo(() => {
    const map = new Map<string, AdminEventReview>();
    for (const row of catalog) {
      if (row.createdByHandle === null && row.createdByName === null) continue;
      map.set(row.id, {
        owner: row.createdByHandle
          ? `@${row.createdByHandle}`
          : (row.createdByName ?? "Photographer"),
        status: row.reviewStatus,
      });
    }
    return map;
  }, [catalog]);

  const liveCount = byState(catalog, "live").length;

  async function handleCreateSubmit(payload: {
    name: string;
    location: string;
    date: string;
    pricePerPhoto: number;
    organizerName: string;
    description: string;
    cover: File | null;
    removeCover: boolean;
  }) {
    try {
      const ok = await createEvent({
        title: payload.name,
        location: payload.location,
        date: payload.date,
        pricePerPhoto: payload.pricePerPhoto,
        organizerName: payload.organizerName,
        description: payload.description,
        cover: payload.cover,
      });
      if (!ok) return false;
      setCreateOpen(false);
      showToast({ kind: "success", message: `Created · ${payload.name}` });
      return true;
    } catch {
      // The domain hook reports errors; retain the form for retry.
      return false;
    }
  }

  async function handleEditSubmit(payload: {
    name: string;
    location: string;
    date: string;
    pricePerPhoto: number;
    organizerName: string;
    description: string;
    cover: File | null;
    removeCover: boolean;
  }) {
    if (!editTarget) return false;
    try {
      // Only forward pricePerPhoto when it actually changed from the
      // prefilled value — keeps the PATCH payload minimal and avoids
      // logging a no-op price entry into events.admin_overrides JSONB.
      const priceChanged =
        editTarget.pricePerPhoto === undefined ||
        editTarget.pricePerPhoto !== payload.pricePerPhoto;
      // Same minimal-payload treatment for organizer + notes.
      const organizerChanged =
        (editTarget.organizerName ?? "") !== payload.organizerName;
      const descriptionChanged =
        (editTarget.description ?? "") !== payload.description;
      const ok = await editEvent(editTarget.id, {
        expectedRevision: editTarget.reviewRevision,
        title: payload.name,
        location: payload.location,
        date: payload.date,
        ...(priceChanged ? { pricePerPhoto: payload.pricePerPhoto } : {}),
        ...(organizerChanged ? { organizerName: payload.organizerName } : {}),
        ...(descriptionChanged ? { description: payload.description } : {}),
        cover: payload.cover,
        removeCover: payload.removeCover,
      });
      if (!ok) return false;
      setEditOpen(false);
      showToast({ kind: "success", message: `Updated · ${payload.name}` });
      return true;
    } catch {
      // The domain hook reports errors; retain the form for retry.
      return false;
    }
  }

  async function handleDelete(event: ListEvent) {
    await confirm({
      title: "Delete event?",
      message: (
        <>
          Removes <strong className="text-ink">{event.name}</strong> from{" "}
          <span className="font-mono text-[13px]">/events</span>, the
          photographer upload picker, and the admin board.
        </>
      ),
      confirmLabel: "Delete event",
      danger: true,
      onConfirm: async () => {
        const ok = await deleteEvent(event.id);
        if (!ok) throw new Error("Another action is pending or the session changed. Review this event before retrying.");
        showToast({ kind: "info", message: `Deleted · ${event.name}` });
      },
    });
  }

  return (
    <>
      <Header
        total={catalog.length}
        liveCount={liveCount}
        onCreate={() => setCreateOpen(true)}
      />
      {query.isPending && <p role="status" className="font-sans text-sm text-slate">Loading events…</p>}
      {query.isError && (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-error">Couldn&apos;t load events. Please try again.</p>
          <button type="button" onClick={() => void query.refetch()} disabled={query.isFetching} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry</button>
        </div>
      )}
      {query.data && grouped.map((section) => {
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
                    review={reviewById.get(event.id)}
                    pendingLabel={pendingAction(event.id)}
                    onEdit={(e) => { if (editTarget?.id !== e.id || !recoveryNotice(e.id)) setEditTarget(e); setEditOpen(true); }}
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
        pending={isPending("create")}
        recoveryNotice={recoveryNotice("create")}
        onClose={() => setCreateOpen(false)}
        onSubmit={handleCreateSubmit}
      />
      <AdminEventFormModal
        open={editOpen}
        mode="edit"
        pending={!!editTarget && isPending(editTarget.id)}
        recoveryNotice={editTarget ? recoveryNotice(editTarget.id) : null}
        event={editTarget}
        onClose={() => setEditOpen(false)}
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
          <Kicker as="p">
            Events ·{" "}
            <span className="tnum">{total.toLocaleString()}</span> on platform ·{" "}
            <span className="tnum">{liveCount}</span> live
          </Kicker>
          <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
            Events.
          </h1>
        </div>
        <button
          type="button"
          onClick={onCreate}
          className={cn(BTN_PRIMARY, BTN_SIZE.sm, "shrink-0")}
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
