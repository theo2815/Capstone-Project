"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { AdminFlagCard } from "@/components/admin/admin-flag-card";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import { AdminFlagHideModal } from "@/components/admin/admin-flag-hide-modal";
import { AdminEscalateModal } from "@/components/admin/admin-escalate-modal";
import {
  useAdminFlagStore,
  mergeFlagsWithOverrides,
} from "@/store/admin-flag-store";
import { useAdminFlags, EMPTY_FLAGS } from "@/hooks/use-admin-data";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import {
  useDrawerVerbs,
  useQueueKeyboardNav,
  useSearchFocusShortcut,
} from "@/hooks/use-admin-keyboard";
import { type Flag, type FlagStatus, FLAG_REASON_LABEL } from "@/lib/admin-flags";
import { getEventById } from "@/lib/event-catalog";
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { formatLongDate } from "@/lib/format";

const STATUS_LABEL: Record<FlagStatus, string> = {
  open: "Open",
  hidden: "Hidden",
  dismissed: "Dismissed",
  escalated: "Escalated",
};

function statusTone(status: FlagStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "hidden":
      return "ink";
    case "dismissed":
      return "slate";
    case "escalated":
      return "muted";
  }
}

// Flags queue body — extracted from /admin/flags/page.tsx so the
// universal /admin/inbox can render the same search-filtered four slabs.
// Search state stays internal to the queue (no need to lift it to the
// page or URL — query is a transient scan tool, not a deep-link target).
//
// Phase 3 adds the focus-mode drawer. Card click → drawer with the photo
// at full width, the report, reviewer-side metadata, and Hide/Dismiss/
// Escalate in the footer. URL state via ?row=<flagId>. Inline card
// actions stay so users can triage without leaving the list. Queue-level
// modal mounts (drawer-side) keep the ESC chain clean — the drawer's
// escDisabled stands down whenever any of them is open.

export function FlagsQueue() {
  const apiFlags = useAdminFlags();
  const overrides = useAdminFlagStore((s) => s.overrides);
  const hide = useAdminFlagStore((s) => s.hide);
  const dismiss = useAdminFlagStore((s) => s.dismiss);
  const escalate = useAdminFlagStore((s) => s.escalate);
  const { showToast } = useToast();

  const [query, setQuery] = useState("");
  const [rowId, setRowId] = useUrlState<string>("row", "");
  const [drawerHideOpen, setDrawerHideOpen] = useState(false);
  const [drawerEscalateOpen, setDrawerEscalateOpen] = useState(false);
  const [drawerConfirmDismiss, setDrawerConfirmDismiss] = useState(false);

  const effective = useMemo(
    () => mergeFlagsWithOverrides(apiFlags ?? EMPTY_FLAGS, overrides),
    [apiFlags, overrides],
  );

  const byId = useMemo(() => {
    const map = new Map<string, Flag>();
    for (const f of effective) map.set(f.id, f);
    return map;
  }, [effective]);

  const openRow = useMemo<Flag | null>(() => {
    if (!rowId) return null;
    return byId.get(rowId) ?? null;
  }, [rowId, byId]);

  useEffect(() => {
    if (rowId && !openRow) setRowId("");
  }, [rowId, openRow, setRowId]);

  // Reset the dismiss-confirm two-step whenever the open row changes (or
  // closes), so a stale "Confirm dismiss" label never carries over to a
  // different flag.
  useEffect(() => {
    setDrawerConfirmDismiss(false);
  }, [rowId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length === 0) return effective;
    return effective.filter((f) => {
      return (
        FLAG_REASON_LABEL[f.reason].toLowerCase().includes(q) ||
        f.photographerHandle.toLowerCase().includes(q) ||
        f.reportedBy.toLowerCase().includes(q) ||
        f.note.toLowerCase().includes(q)
      );
    });
  }, [effective, query]);

  const open = useMemo(
    () =>
      [...filtered.filter((f) => f.status === "open")].sort((a, b) =>
        a.reportedAt.localeCompare(b.reportedAt),
      ),
    [filtered],
  );
  const hidden = useMemo(
    () =>
      [...filtered.filter((f) => f.status === "hidden")].sort((a, b) =>
        (b.reviewedAt ?? "").localeCompare(a.reviewedAt ?? ""),
      ),
    [filtered],
  );
  const dismissed = useMemo(
    () =>
      [...filtered.filter((f) => f.status === "dismissed")].sort((a, b) =>
        (b.reviewedAt ?? "").localeCompare(a.reviewedAt ?? ""),
      ),
    [filtered],
  );
  const escalated = useMemo(
    () =>
      [...filtered.filter((f) => f.status === "escalated")].sort((a, b) =>
        b.reportedAt.localeCompare(a.reportedAt),
      ),
    [filtered],
  );

  const drawerEscDisabled = drawerHideOpen || drawerEscalateOpen;

  // Pagination — load-more model. Open keeps full visibility — that's
  // the actionable triage list. Read-only history slabs start at 10 and
  // grow by 10 per click. A search-query change resets all three loaded
  // counts so the user doesn't carry a long load over a narrow filter.
  const [hiddenLoaded, setHiddenLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const [dismissedLoaded, setDismissedLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const [escalatedLoaded, setEscalatedLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  useEffect(() => {
    setHiddenLoaded(PAGE_SIZE.ADMIN_INITIAL);
    setDismissedLoaded(PAGE_SIZE.ADMIN_INITIAL);
    setEscalatedLoaded(PAGE_SIZE.ADMIN_INITIAL);
  }, [query]);
  const hiddenVisible = useMemo(
    () => hidden.slice(0, hiddenLoaded),
    [hidden, hiddenLoaded],
  );
  const dismissedVisible = useMemo(
    () => dismissed.slice(0, dismissedLoaded),
    [dismissed, dismissedLoaded],
  );
  const escalatedVisible = useMemo(
    () => escalated.slice(0, escalatedLoaded),
    [escalated, escalatedLoaded],
  );

  // Keyboard nav iterates every flag slab in render order. The drawer
  // wires only the H verb (Hide…) — Dismiss has no corresponding verb in
  // the E/R/H/S whitelist and Escalate is a tertiary action. `/` focuses
  // the search input from anywhere outside the drawer. Use the visible
  // slices for paginated slabs so J/K only visits rows currently
  // rendered.
  const rowIds = useMemo(
    () =>
      [...open, ...hiddenVisible, ...dismissedVisible, ...escalatedVisible].map(
        (f) => f.id,
      ),
    [open, hiddenVisible, dismissedVisible, escalatedVisible],
  );
  const queueNavDisabled = openRow !== null || drawerEscDisabled;
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: queueNavDisabled,
    onActivate: (id) => {
      setFocusedId(id);
      setRowId(id);
    },
  });
  useEffect(() => {
    if (rowId) setFocusedId(rowId);
  }, [rowId, setFocusedId]);

  function handleOpenRow(id: string) {
    setFocusedId(id);
    setRowId(id);
  }

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  useSearchFocusShortcut({
    active: openRow === null && !drawerEscDisabled,
    inputRef: searchInputRef,
  });

  function handleDrawerHideSubmit(reason: string | null) {
    if (!openRow) return;
    hide(openRow.id, reason);
    setDrawerHideOpen(false);
    showToast({ kind: "info", message: `Hidden · ${openRow.id}` });
    setRowId("");
  }

  function handleDrawerDismissConfirm() {
    if (!openRow) return;
    dismiss(openRow.id);
    setDrawerConfirmDismiss(false);
    showToast({ kind: "info", message: `Dismissed · ${openRow.id}` });
    setRowId("");
  }

  function handleDrawerEscalateSubmit(note: string | null) {
    if (!openRow) return;
    escalate(openRow.id, note);
    setDrawerEscalateOpen(false);
    showToast({ kind: "info", message: `Escalated · ${openRow.id}` });
    setRowId("");
  }

  // Only register H when the open flag is actionable (status === "open")
  // and no stacked modal is taking the keystroke instead.
  useDrawerVerbs({
    active:
      openRow !== null &&
      openRow.status === "open" &&
      !drawerHideOpen &&
      !drawerEscalateOpen,
    verbs: openRow
      ? {
          H: () => setDrawerHideOpen(true),
        }
      : {},
  });

  return (
    <>
      <div className="mt-8 md:mt-10">
        <label htmlFor="flag-search" className="sr-only">
          Search flags
        </label>
        <input
          id="flag-search"
          ref={searchInputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by reason, photographer, reporter, or note  ·  /"
          className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
        />
        {query.trim().length > 0 && (
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-3 tnum">
            {filtered.length} match{filtered.length === 1 ? "" : "es"}
          </p>
        )}
      </div>

      <FlagSlab
        id="open"
        number="01"
        title="Open"
        caption="Awaiting decision"
        totalCount={open.length}
        rows={open}
        empty="Queue is clear — no open flags."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        primaryFirst
      />
      <FlagSlab
        id="hidden"
        number="02"
        title="Hidden"
        caption="Photo removed from runner views"
        totalCount={hidden.length}
        rows={hiddenVisible}
        empty="No hidden photos."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: hiddenVisible.length,
          total: hidden.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setHiddenLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <FlagSlab
        id="dismissed"
        number="03"
        title="Dismissed"
        caption="False alarms"
        totalCount={dismissed.length}
        rows={dismissedVisible}
        empty="No dismissed flags."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: dismissedVisible.length,
          total: dismissed.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setDismissedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <FlagSlab
        id="escalated"
        number="04"
        title="Escalated"
        caption="Pushed to higher review"
        totalCount={escalated.length}
        rows={escalatedVisible}
        empty="No escalations."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: escalatedVisible.length,
          total: escalated.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setEscalatedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          escDisabled={drawerEscDisabled}
          kicker={`Flag · ${openRow.id}`}
          title={FLAG_REASON_LABEL[openRow.reason]}
          rightHeader={
            <AdminStatusPill
              label={STATUS_LABEL[openRow.status]}
              tone={statusTone(openRow.status)}
            />
          }
          subtitle={<FlagSubtitle flag={openRow} />}
          actions={
            <FlagDrawerActions
              flag={openRow}
              confirmDismiss={drawerConfirmDismiss}
              onHide={() => setDrawerHideOpen(true)}
              onToggleConfirmDismiss={() =>
                setDrawerConfirmDismiss((v) => !v)
              }
              onConfirmDismiss={handleDrawerDismissConfirm}
              onEscalate={() => setDrawerEscalateOpen(true)}
            />
          }
        >
          <FlagDetailBody flag={openRow} />
        </AdminDetailDrawer>
      )}

      {openRow && (
        <>
          <AdminFlagHideModal
            open={drawerHideOpen}
            onClose={() => setDrawerHideOpen(false)}
            onSubmit={handleDrawerHideSubmit}
            flagId={openRow.id}
          />
          <AdminEscalateModal
            open={drawerEscalateOpen}
            onClose={() => setDrawerEscalateOpen(false)}
            onSubmit={handleDrawerEscalateSubmit}
            targetLabel={`Escalate ${openRow.id}`}
            body="Push this flag to the next review tier. The photo stays live in the meantime; higher tier decides hide vs. dismiss."
          />
        </>
      )}
    </>
  );
}

export function useOpenFlagsCount(): number {
  const apiFlags = useAdminFlags();
  const overrides = useAdminFlagStore((s) => s.overrides);
  return useMemo(
    () =>
      mergeFlagsWithOverrides(apiFlags ?? EMPTY_FLAGS, overrides).filter(
        (f) => f.status === "open",
      ).length,
    [apiFlags, overrides],
  );
}

export function useHiddenFlagsCount(): number {
  const apiFlags = useAdminFlags();
  const overrides = useAdminFlagStore((s) => s.overrides);
  return useMemo(
    () =>
      mergeFlagsWithOverrides(apiFlags ?? EMPTY_FLAGS, overrides).filter(
        (f) => f.status === "hidden",
      ).length,
    [apiFlags, overrides],
  );
}

function FlagSlab({
  id,
  number,
  title,
  caption,
  rows,
  totalCount,
  empty,
  onOpenRow,
  focusedId,
  primaryFirst,
  loadMore,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  rows: Flag[];
  totalCount: number;
  empty: string;
  onOpenRow: (id: string) => void;
  focusedId: string | null;
  primaryFirst?: boolean;
  loadMore?: {
    shown: number;
    total: number;
    increment: number;
    onLoadMore: () => void;
  };
}) {
  const noun = totalCount === 1 ? "flag" : "flags";
  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${totalCount} ${noun}`}
    >
      {totalCount === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{empty}</p>
      ) : (
        <>
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {rows.map((row, i) => (
              <li key={row.id}>
                <AdminFlagCard
                  flag={row}
                  primary={primaryFirst === true && i === 0}
                  focused={focusedId === row.id}
                  onOpen={() => onOpenRow(row.id)}
                />
              </li>
            ))}
          </ul>
          {loadMore && (
            <LoadMoreButton
              shown={loadMore.shown}
              total={loadMore.total}
              increment={loadMore.increment}
              onLoadMore={loadMore.onLoadMore}
            />
          )}
        </>
      )}
    </Slab>
  );
}

function FlagSubtitle({ flag }: { flag: Flag }) {
  const event = getEventById(flag.eventId);
  const eventName = event?.name ?? `Event ${flag.eventId}`;
  return (
    <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
      {flag.reportedBy === "system" ? "System" : `@${flag.reportedBy}`}
      <span className="text-slate-soft"> · </span>
      @{flag.photographerHandle}
      <span className="text-slate-soft"> · </span>
      {eventName}
      <span className="text-slate-soft"> · </span>
      {formatLongDate(flag.reportedAt)}
    </p>
  );
}

function FlagDrawerActions({
  flag,
  confirmDismiss,
  onHide,
  onToggleConfirmDismiss,
  onConfirmDismiss,
  onEscalate,
}: {
  flag: Flag;
  confirmDismiss: boolean;
  onHide: () => void;
  onToggleConfirmDismiss: () => void;
  onConfirmDismiss: () => void;
  onEscalate: () => void;
}) {
  const isOpen = flag.status === "open";

  if (!isOpen) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        This flag is closed. No further actions are available.
      </p>
    );
  }

  if (confirmDismiss) {
    return (
      <>
        <button
          type="button"
          onClick={onToggleConfirmDismiss}
          className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={onConfirmDismiss}
          className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink border border-line hover:bg-ink hover:text-surface hover:border-ink transition-colors rounded-full px-5 py-2"
        >
          Confirm dismiss
        </button>
      </>
    );
  }

  return (
    <>
      <button
        type="button"
        onClick={onEscalate}
        className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2"
      >
        Escalate…
      </button>
      <button
        type="button"
        onClick={onToggleConfirmDismiss}
        className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2"
      >
        Dismiss
      </button>
      <button
        type="button"
        onClick={onHide}
        className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-surface bg-fresh hover:bg-fresh-deep transition-colors rounded-full px-5 py-2"
      >
        Hide…
      </button>
    </>
  );
}

function FlagDetailBody({ flag }: { flag: Flag }) {
  const event = getEventById(flag.eventId);
  const eventName = event?.name ?? `Event ${flag.eventId}`;
  const cover = syntheticCoverGradient(flag.id);

  return (
    <div className="space-y-10">
      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Photo
        </Kicker>
        {flag.photoSnapshot.thumbnailUrl ? (
          <div className="aspect-[16/9] rounded-2xl border border-line overflow-hidden bg-bone-deep relative">
            <img
              src={flag.photoSnapshot.thumbnailUrl}
              alt={flag.photoSnapshot.alt}
              className="w-full h-full object-cover"
            />
          </div>
        ) : (
          <div
            className="aspect-[16/9] rounded-2xl border border-line"
            style={{
              background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
            }}
            aria-label={flag.photoSnapshot.alt}
          />
        )}
        <p className="font-sans text-sm text-ink-soft mt-3">
          {flag.photoSnapshot.alt}
        </p>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Report
        </Kicker>
        <p className="font-sans text-sm md:text-base text-ink-soft whitespace-pre-line">
          {flag.note}
        </p>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Context
        </Kicker>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow
            label="Reported by"
            value={
              flag.reportedBy === "system" ? "System" : `@${flag.reportedBy}`
            }
            mono
          />
          <FieldRow
            label="Photographer"
            value={`@${flag.photographerHandle}`}
            mono
          />
          <FieldRow label="Event" value={eventName} mono={false} />
          <FieldRow
            label="Reported at"
            value={formatLongDate(flag.reportedAt)}
            mono
          />
          {flag.reviewedAt && (
            <FieldRow
              label="Reviewed at"
              value={formatLongDate(flag.reviewedAt)}
              mono
            />
          )}
        </dl>
      </section>

      {flag.reviewerNote && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Reviewer note
          </Kicker>
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <p className="font-sans text-sm text-ink-soft">
              {flag.reviewerNote}
            </p>
          </div>
        </section>
      )}
    </div>
  );
}

function FieldRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono: boolean;
}) {
  return (
    <div>
      <dt className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}
