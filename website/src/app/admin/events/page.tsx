"use client";

import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import {
  AdminEventCard,
  type AdminEventReview,
} from "@/components/admin/admin-event-card";
import { AdminEventFormModal } from "@/components/admin/admin-event-form-modal";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";
import { useUrlState } from "@/hooks/use-url-state";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { useAdminEventsInfinite, useAdminEventCounts, useAdminEventActions } from "@/hooks/use-admin-events";
import { useToast } from "@/hooks/use-toast";
import { useConfirmation } from "@/hooks/use-confirmation";
import type { EventState, ListEvent } from "@/types/event";

// Admin Events page. Four status tabs (Live / Upcoming / Recent / Archive),
// one list of <AdminEventCard>s at a time. Lifecycle state is derived from `event.date`
// (see `lib/event-catalog.ts`) — admin only edits Title/Location/Date and
// the four states fall out automatically:
//
//   future date            → Upcoming
//   day 0..3 (race + 4d)   → Live  (photographer upload window)
//   day 4..89              → Recent (gallery for sale, no more uploads)
//   day 90+                → Archive

const SECTIONS: ReadonlyArray<{
  id: string;
  title: string;
  caption: string;
  matchState: EventState;
  cols: string;
}> = [
  {
    id: "live",
    title: "Live now",
    caption: "Currently accepting uploads · 4-day grace from race day",
    matchState: "live",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "upcoming",
    title: "Upcoming",
    caption: "Race day pending",
    matchState: "upcoming",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "recent",
    title: "Recent",
    caption: "Galleries open for sale · upload window closed",
    matchState: "open",
    cols: "grid-cols-1 md:grid-cols-2",
  },
  {
    id: "archive",
    title: "Archive",
    caption: "Read-only history",
    matchState: "past",
    cols: "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
  },
];

export default function AdminEventsPage() {
  // Proposal 12: one status tab at a time, kept in the URL so Back returns to it.
  const [tabRaw, setTab] = useUrlState<string>("status", "live");
  const section = SECTIONS.find((x) => x.id === tabRaw) ?? SECTIONS[0];
  // AX-21 (EV1): each tab is its own server list, paged, and the search runs on the server.
  const [search, setSearch] = useUrlState<string>("q", "", { debounceMs: 300 });
  // The server search follows the URL, which trails typing by the 300 ms debounce.
  const q = (useSearchParams().get("q") ?? "").trim();
  const list = useAdminEventsInfinite(section.matchState, q);
  const catalog = list.items;
  const counts = useAdminEventCounts(q);
  const platformCounts = useAdminEventCounts("");
  const { createEvent, editEvent, deleteEvent, setListing, pendingAction, isPending, recoveryNotice } = useAdminEventActions();
  const { showToast } = useToast();
  const { confirm } = useConfirmation();

  const [createOpen, setCreateOpen] = useState(false);
  const [editTarget, setEditTarget] = useState<ListEvent | null>(null);
  const [editOpen, setEditOpen] = useState(false);

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
        visibility: row.visibility,
        listed: row.listedOnEvents,
      });
    }
    return map;
  }, [catalog]);

  // AA-R17b: shared races (typed) → hidden or not; the listing switch hides or relists them.
  // Not a draft: relisting marks a race admin approved, which only a review may do.
  const raceHiddenById = useMemo(
    () => new Map(catalog.filter((row) => row.eventType && row.status !== "DRAFT").map((row) => [row.id, !!row.hiddenAt])),
    [catalog],
  );

  async function handleRaceListing(event: ListEvent, listed: boolean) {
    await confirm({
      title: listed ? "Relist this race?" : "Hide this race?",
      message: listed
        ? <>Puts <strong className="text-ink">{event.name}</strong> back on /events and the upload pickers, and marks it as checked by an admin.</>
        : <>Takes <strong className="text-ink">{event.name}</strong> off /events and the upload pickers. Its link still opens, and it still shows on its photographers&apos; public profiles.</>,
      confirmLabel: listed ? "Relist race" : "Hide race",
      danger: !listed,
      onConfirm: async () => {
        const ok = await setListing(event.id, listed);
        if (!ok) throw new Error("Another action is pending or the session changed. Review this race before retrying.");
        showToast({ kind: "success", message: `${listed ? "Relisted" : "Hidden"} · ${event.name}` });
      },
    });
  }

  // EV5: "Show on /events" asks first, as Relist does; the box follows the server.
  async function handleSetListing(event: ListEvent, listed: boolean) {
    await confirm({
      title: listed ? "Show this event on /events?" : "Take this event off /events?",
      message: listed
        ? <>Puts <strong className="text-ink">{event.name}</strong> on the public /events list.</>
        : <>Takes <strong className="text-ink">{event.name}</strong> off the public /events list. Its link still opens.</>,
      confirmLabel: listed ? "Show on /events" : "Take off /events",
      onConfirm: async () => {
        const ok = await setListing(event.id, listed);
        if (!ok) throw new Error("Another action is pending or the session changed. Review this event before retrying.");
      },
    });
  }

  async function handleCreateSubmit(payload: {
    name: string;
    location: string;
    regionCode: string | null;
    provinceCode: string | null;
    date: string;
    pricePerPhoto: number;
    organizerName: string;
    description: string;
    cover: File | null;
    removeCover: boolean;
    eventType?: string;
  }) {
    try {
      const ok = await createEvent({
        eventType: payload.eventType,
        title: payload.name,
        location: payload.location,
        regionCode: payload.regionCode ?? undefined,
        provinceCode: payload.provinceCode ?? undefined,
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
    regionCode: string | null;
    provinceCode: string | null;
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
      const pairChanged = (editTarget.regionCode ?? null) !== payload.regionCode ||
        (editTarget.provinceCode ?? null) !== payload.provinceCode;
      const ok = await editEvent(editTarget.id, {
        expectedRevision: editTarget.reviewRevision,
        title: payload.name,
        location: payload.location,
        ...(pairChanged && payload.regionCode && payload.provinceCode
          ? { regionCode: payload.regionCode, provinceCode: payload.provinceCode } : {}),
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
        counts={platformCounts}
        onCreate={() => setCreateOpen(true)}
      />
      <div className="mt-8 md:mt-10">
        <label htmlFor="event-search" className="sr-only">Search events</label>
        <input
          id="event-search"
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by event name or place"
          className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
        />
      </div>
      {/* No count when the counts route is missing or loading: a 0 would read "nothing here". */}
      <AdminStatusTabs
        label="Event status"
        tabs={SECTIONS.map((g) => ({ key: g.id, label: g.title, count: counts?.[g.matchState] }))}
        value={section.id}
        onChange={setTab}
      />
      {list.isLoading && (
        <div className="mt-8 md:mt-10 space-y-3" aria-busy="true">
          {Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-20 w-full rounded-2xl" />)}
        </div>
      )}
      {!!list.error && (
        <div role="alert" className="mt-6 space-y-3">
          <p className="font-sans text-sm text-error">Couldn&apos;t load events. Please try again.</p>
          <button type="button" onClick={list.refetch} disabled={list.isFetching} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry</button>
        </div>
      )}
      {/* EV6: the tab already names the status and its count; only the caption stays. */}
      {!list.isLoading && !(list.error && catalog.length === 0) && (
          <section key={section.id} id={section.id} aria-labelledby={`${section.id}-heading`} className="pt-6 md:pt-8">
            <h2 id={`${section.id}-heading`} className="sr-only">{section.title}</h2>
            <p className="font-sans text-sm text-slate mb-4">{section.caption}</p>
            {catalog.length === 0 ? (
              <p className="font-sans text-sm text-slate-soft">
                {q ? "No events in this tab match this search." : emptyCopyFor(section.matchState)}
              </p>
            ) : (
              <ul className="border border-line rounded-2xl bg-surface divide-y divide-line overflow-hidden">
                {catalog.map((event) => (
                  <li key={event.id}>
                  <AdminEventCard
                    event={event}
                    review={reviewById.get(event.id)}
                    pendingLabel={pendingAction(event.id)}
                    onEdit={(e) => { if (editTarget?.id !== e.id || !recoveryNotice(e.id)) setEditTarget(e); setEditOpen(true); }}
                    onDelete={handleDelete}
                    onSetListing={handleSetListing}
                    raceHidden={raceHiddenById.get(event.id)}
                    onRaceListing={handleRaceListing}
                  />
                  </li>
                ))}
              </ul>
            )}
            <LoadMoreButton shown={catalog.length} total={list.total} increment={50} isLoading={list.isFetchingNextPage} onLoadMore={list.fetchNextPage} />
          </section>
      )}

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
  counts,
  onCreate,
}: {
  counts: Record<EventState, number> | undefined;
  onCreate: () => void;
}) {
  const total = counts ? Object.values(counts).reduce((n, c) => n + c, 0) : 0;
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <div className="flex items-start justify-between gap-4">
        <div>
          {/* Totals come from the counts route; without it (an older backend) the kicker names the page only. */}
          <Kicker as="p">
            Events
            {counts && (
              <>
                {" "}·{" "}<span className="tnum">{total.toLocaleString()}</span> on platform ·{" "}
                <span className="tnum">{counts.live.toLocaleString()}</span> live
              </>
            )}
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
        Every race on the platform. Photographers can upload for 4 days from
        race day.
      </p>
    </header>
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
