"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { AdminFlagCard, FlagStatusPill, type FlagCardActions } from "@/components/admin/admin-flag-card";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { AdminFlagHideModal } from "@/components/admin/admin-flag-hide-modal";
import { AdminEscalateModal } from "@/components/admin/admin-escalate-modal";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useSearchParams } from "next/navigation";
import {
  useAdminFlagById,
  useAdminFlagHistory,
  useAdminFlags,
  useAdminFlagsQuery,
  useFlagActions,
  EMPTY_FLAGS,
} from "@/hooks/use-admin-data";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import {
  useDrawerVerbs,
  useQueueKeyboardNav,
  useSearchFocusShortcut,
} from "@/hooks/use-admin-keyboard";
import {
  type Flag,
  canDismiss,
  canEscalate,
  canConfirmHold,
  canHide,
  canResolve,
  flagEventName,
  flagReasonLabel,
  isRelevanceFlag,
  relevanceConfirmText,
  serverMessage,
  type FlagConfirmKind,
} from "@/lib/admin-flags";
import { FlagRelevanceEvidence } from "@/components/admin/flag-relevance-evidence";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import { formatLongDate } from "@/lib/format";

const DONE_LABEL = {
  hide: "Hidden",
  dismiss: "Dismissed",
  escalate: "Escalated",
  hold: "Payouts on hold",
  resolve: "Resolved",
} as const;

const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm}`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} px-3`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm}`;

type FlagModal = { kind: "hide" | "escalate" | null; id: string };

// Flags queue body — extracted from /admin/flags/page.tsx so the
// universal /admin/inbox can render the same search-filtered four slabs.
// The search lives in the URL (?q=) so a missing deep link can hand off
// to a search for its id (FLG1).
//
// Phase 3 adds the focus-mode drawer. Card click → drawer with the photo
// at full width, the report, reviewer-side metadata, and Hide/Dismiss/
// Escalate in the footer. URL state via ?row=<flagId>. Inline card
// actions stay so users can triage without leaving the list. Queue-level
// modal mounts (drawer-side) keep the ESC chain clean — the drawer's
// escDisabled stands down whenever any of them is open.

export function FlagsQueue() {
  // Open flags are fetched on their own so the actionable list is never
  // pushed off the end of the newest-first 200-row page by closed history.
  const openQuery = useAdminFlagsQuery({ status: "open" });
  const openPage = openQuery.data ?? null;
  const { hide, dismiss, escalate, resolve, confirmHold, isPending, pendingAction } = useFlagActions();
  const { showToast } = useToast();

  const [query, setQuery] = useUrlState<string>("q", "", { debounceMs: 300 });
  // AX-33 (FLG3): each history slab pages its own server list with the server
  // search (reason, note, handles, reporter name, flag id). The search follows
  // the URL, which trails typing by the 300 ms debounce.
  const q = (useSearchParams().get("q") ?? "").trim();
  const hiddenList = useAdminFlagHistory("hidden", q);
  const dismissedList = useAdminFlagHistory("dismissed", q);
  const resolvedList = useAdminFlagHistory("resolved", q);
  const escalatedList = useAdminFlagHistory("escalated", q);
  const historyLists = [hiddenList, dismissedList, resolvedList, escalatedList];
  const [rowId, setRowId] = useUrlState<string>("row", "");
  const submitting = isPending(rowId);
  // One hide and one escalate modal for the whole queue (FLG6). The id stays
  // after close so the title does not blank while the modal animates out.
  const [modal, setModal] = useState<FlagModal>({ kind: null, id: "" });
  const closeModal = () => setModal((m) => ({ ...m, kind: null }));
  const [drawerConfirm, setDrawerConfirm] = useState<FlagConfirmKind | null>(null);
  const [missingId, setMissingId] = useState("");

  // Open stays one list, searched here, so the actionable flags show in full.
  const open = useMemo(() => {
    const needle = q.toLowerCase();
    return (openPage?.items ?? EMPTY_FLAGS)
      .filter((f) => needle.length === 0 ||
        f.id.toLowerCase().includes(needle) ||
        flagReasonLabel(f.reason).toLowerCase().includes(needle) ||
        f.photographerHandle.toLowerCase().includes(needle) ||
        f.reportedBy.toLowerCase().includes(needle) ||
        f.note.toLowerCase().includes(needle))
      .sort((a, b) => a.reportedAt.localeCompare(b.reportedAt));
  }, [openPage, q]);
  const hidden = hiddenList.items;
  // Dismissed and resolved share a slab: dismissed pages first, then resolved.
  const dismissed = dismissedList.hasNextPage
    ? dismissedList.items
    : [...dismissedList.items, ...resolvedList.items];
  const dismissedTotal = dismissedList.total + resolvedList.total;
  const escalated = escalatedList.items;

  const listRow = rowId
    ? [openPage?.items ?? EMPTY_FLAGS, ...historyLists.map((l) => l.items)].flat().find((f) => f.id === rowId) ?? null
    : null;
  // A flag past the loaded pages is read by id, so its link still opens.
  const rowQuery = useAdminFlagById(listRow ? "" : rowId);
  const openRow: Flag | null = listRow ?? rowQuery.data ?? null;

  // Clear a ?row= only once every list and the by-id read loaded and none has it (FLG1).
  // The notice is for links that never matched, not a drawer whose flag left.
  // Not while refetching: a stale cached list may not hold a new flag yet.
  const listsLoaded =
    openQuery.isSuccess && !openQuery.isFetching &&
    historyLists.every((l) => !l.isLoading && !l.error && !l.isFetching) &&
    rowQuery.isSuccess && !rowQuery.isFetching;
  const shownRowRef = useRef("");
  if (openRow) shownRowRef.current = openRow.id;
  useEffect(() => {
    if (rowId && !openRow && listsLoaded) {
      if (shownRowRef.current !== rowId) setMissingId(rowId);
      shownRowRef.current = "";
      setRowId("");
    }
  }, [rowId, openRow, listsLoaded, setRowId]);

  // Reset the dismiss-confirm two-step whenever the open row changes (or
  // closes), so a stale "Confirm dismiss" label never carries over to a
  // different flag.
  useEffect(() => {
    setDrawerConfirm(null);
  }, [rowId]);

  const drawerEscDisabled = submitting || modal.kind !== null;

  // Keyboard nav iterates every flag slab in render order. The drawer
  // wires only the H verb (Hide…) — Dismiss has no corresponding verb in
  // the E/R/H/S whitelist and Escalate is a tertiary action. `/` focuses
  // the search input from anywhere outside the drawer. Only rendered rows.
  const rowIds = [...open, ...hidden, ...dismissed, ...escalated].map((f) => f.id);
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
    setMissingId("");
    setFocusedId(id);
    setRowId(id);
  }

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  useSearchFocusShortcut({
    active: openRow === null && !drawerEscDisabled,
    inputRef: searchInputRef,
  });

  // One runner for card and drawer actions: success toast, or the server's reason.
  async function runAction(
    verb: keyof typeof DONE_LABEL,
    id: string,
    action: () => Promise<boolean>,
  ): Promise<boolean> {
    try {
      if (!(await action())) return false;
      showToast({ kind: "success", message: `${DONE_LABEL[verb]} · ${id}` });
      // A hold leaves the flag open: stay on it to hide the batch next.
      if (verb !== "hold" && id === rowId) setRowId("");
      return true;
    } catch (error) {
      showToast({
        kind: "error",
        message:
          serverMessage(error) ??
          `Could not ${verb} ${id} — check the current status before retrying.`,
      });
      return false;
    }
  }

  function confirmAction(id: string, kind: FlagConfirmKind) {
    if (kind === "hold") return runAction("hold", id, () => confirmHold(id));
    if (kind === "resolve") return runAction("resolve", id, () => resolve(id));
    return runAction("dismiss", id, () => dismiss(id));
  }

  const cardActions: FlagCardActions = {
    isPending,
    pendingAction,
    onHide: (id) => setModal({ kind: "hide", id }),
    onEscalate: (id) => setModal({ kind: "escalate", id }),
    onConfirm: confirmAction,
  };

  async function handleHideSubmit(reason: string | null) {
    const { id } = modal;
    if (await runAction("hide", id, () => hide(id, reason))) closeModal();
  }

  async function handleEscalateSubmit(note: string | null) {
    const { id } = modal;
    if (await runAction("escalate", id, () => escalate(id, note))) closeModal();
  }

  async function handleDrawerConfirm() {
    if (!openRow || !drawerConfirm) return;
    if (await confirmAction(openRow.id, drawerConfirm)) setDrawerConfirm(null);
  }

  // Only register H when the flag can still be hidden (open or escalated)
  // and no stacked modal is taking the keystroke instead.
  useDrawerVerbs({
    active:
      openRow !== null &&
      canHide(openRow) &&
      !submitting &&
      modal.kind === null,
    verbs: openRow
      ? {
          H: () => setModal({ kind: "hide", id: openRow.id }),
        }
      : {},
  });

  const loadFailed = openQuery.isError || historyLists.some((l) => !!l.error);
  // A failed list with no rows is not ready: it must never read as an empty slab (FLG2).
  const ready = openPage !== null && historyLists.every((l) => !l.isLoading && (!l.error || l.items.length > 0));
  const matchCount = open.length + hiddenList.total + dismissedTotal + escalatedList.total;
  const pager = (list: typeof hiddenList, shown: number, total = list.total) => ({
    shown,
    total,
    isLoading: list.isFetchingNextPage,
    onLoadMore: list.fetchNextPage,
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
          onChange={(e) => { setQuery(e.target.value); setMissingId(""); }}
          placeholder="Search by reason, photographer, reporter, or note  ·  /"
          className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
        />
        {q.length > 0 && ready && (
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-3 tnum">
            {matchCount} match{matchCount === 1 ? "" : "es"}
          </p>
        )}
        {missingId && (
          <div role="status" className="mt-6 space-y-3">
            <p className="font-sans text-sm text-ink-soft">This flag is not in the open list.</p>
            <button
              type="button"
              onClick={() => { setQuery(missingId); setMissingId(""); }}
              className={secondaryBtn}
            >
              Search all flags for {missingId}
            </button>
          </div>
        )}
        {loadFailed && (
          <div role="alert" className="mt-6 space-y-3">
            <p className="font-sans text-sm text-error">Couldn&apos;t load flags. Please try again.</p>
            <button
              type="button"
              onClick={() => { void openQuery.refetch(); historyLists.forEach((l) => l.refetch()); }}
              disabled={openQuery.isFetching || historyLists.some((l) => l.isFetching)}
              className={secondaryBtn}
            >
              Retry
            </button>
          </div>
        )}
      </div>

      {!ready && !loadFailed && (
        <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-5" aria-busy="true">
          {[0, 1].map((i) => <Skeleton key={i} className="h-72 w-full rounded-2xl" />)}
        </div>
      )}

      {ready && <>
      <FlagSlab
        id="open"
        number="01"
        title="Open"
        caption="Awaiting decision"
        totalCount={open.length}
        rows={open}
        empty="Queue is clear — no open flags."
        onOpenRow={handleOpenRow}
        actions={cardActions}
        focusedId={focusedId}
        primaryFirst
      />
      <FlagSlab
        id="hidden"
        number="02"
        title="Hidden"
        caption="Photo removed from runner views"
        totalCount={hiddenList.total}
        rows={hidden}
        empty="No hidden photos."
        onOpenRow={handleOpenRow}
        actions={cardActions}
        focusedId={focusedId}
        loadMore={pager(hiddenList, hidden.length)}
      />
      <FlagSlab
        id="dismissed"
        number="03"
        title="Dismissed"
        caption="False alarms"
        totalCount={dismissedTotal}
        rows={dismissed}
        empty="No dismissed flags."
        onOpenRow={handleOpenRow}
        actions={cardActions}
        focusedId={focusedId}
        loadMore={pager(dismissedList.hasNextPage ? dismissedList : resolvedList, dismissed.length, dismissedTotal)}
      />
      <FlagSlab
        id="escalated"
        number="04"
        title="Escalated"
        caption="Pushed to higher review"
        totalCount={escalatedList.total}
        rows={escalated}
        empty="No escalations."
        onOpenRow={handleOpenRow}
        actions={cardActions}
        focusedId={focusedId}
        loadMore={pager(escalatedList, escalated.length)}
      />
      </>}

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          escDisabled={drawerEscDisabled}
          kicker={`Flag · ${openRow.id}`}
          title={flagReasonLabel(openRow.reason)}
          rightHeader={<FlagStatusPill status={openRow.status} />}
          subtitle={<FlagSubtitle flag={openRow} />}
          actions={
            <FlagDrawerActions
              submitting={submitting}
              flag={openRow}
              confirm={drawerConfirm}
              onHide={() => cardActions.onHide(openRow.id)}
              onToggleConfirm={(kind) =>
                setDrawerConfirm((c) => (c === kind ? null : kind))
              }
              onConfirm={handleDrawerConfirm}
              onEscalate={() => cardActions.onEscalate(openRow.id)}
            />
          }
        >
          {submitting && <p role="status" className="font-sans text-sm text-slate">{pendingAction(rowId)}</p>}
          <FlagDetailBody flag={openRow} />
        </AdminDetailDrawer>
      )}

      <AdminFlagHideModal
        submitting={isPending(modal.id)}
        open={modal.kind === "hide"}
        onClose={closeModal}
        onSubmit={handleHideSubmit}
        flagId={modal.id}
      />
      <AdminEscalateModal
        submitting={isPending(modal.id)}
        open={modal.kind === "escalate"}
        onClose={closeModal}
        onSubmit={handleEscalateSubmit}
        targetLabel={`Escalate ${modal.id}`}
        body="Push this flag to the next review tier. The photo stays live in the meantime; higher tier decides hide vs. dismiss."
      />
    </>
  );
}

// null until loaded, so a failed read never shows "0 open" (FLG2).
export function useOpenFlagsCount(): number | null {
  return useAdminFlags({ status: "open" })?.total ?? null;
}

export function useHiddenFlagsCount(): number | null {
  return useAdminFlags({ status: "hidden" })?.total ?? null;
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
  actions,
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
  actions: FlagCardActions;
  focusedId: string | null;
  primaryFirst?: boolean;
  loadMore?: {
    shown: number;
    total: number;
    isLoading: boolean;
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
                  actions={actions}
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
              increment={50}
              isLoading={loadMore.isLoading}
              onLoadMore={loadMore.onLoadMore}
            />
          )}
        </>
      )}
    </Slab>
  );
}

function FlagSubtitle({ flag }: { flag: Flag }) {
  const eventName = flagEventName(flag);
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
  submitting,
  flag,
  confirm,
  onHide,
  onToggleConfirm,
  onConfirm,
  onEscalate,
}: {
  submitting: boolean;
  flag: Flag;
  confirm: FlagConfirmKind | null;
  onHide: () => void;
  onToggleConfirm: (kind: FlagConfirmKind) => void;
  onConfirm: () => void;
  onEscalate: () => void;
}) {
  const relevance = isRelevanceFlag(flag);
  if (!canDismiss(flag)) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        This flag is closed. No further actions are available.
      </p>
    );
  }

  if (confirm) {
    return (
      <>
        {(relevance || confirm === "hold") && (
          <p className="basis-full text-right font-sans text-sm text-ink-soft">
            {relevanceConfirmText(confirm)}
          </p>
        )}
        <button
          type="button" disabled={submitting}
          onClick={() => onToggleConfirm(confirm)}
          className={tertiaryBtn}
        >
          Cancel
        </button>
        <button
          type="button" disabled={submitting}
          onClick={onConfirm}
          className={secondaryBtn}
        >
          {confirm === "hold"
            ? "Confirm hold"
            : confirm === "resolve"
              ? "Confirm resolve"
              : flag.status === "hidden"
                ? "Confirm restore"
                : "Confirm dismiss"}
        </button>
      </>
    );
  }

  return (
    <>
      {canEscalate(flag) && (
        <button
          type="button" disabled={submitting}
          onClick={onEscalate}
          className={tertiaryBtn}
        >
          Escalate…
        </button>
      )}
      {canResolve(flag) && (
        <button
          type="button" disabled={submitting}
          onClick={() => onToggleConfirm("resolve")}
          className={tertiaryBtn}
        >
          Resolve
        </button>
      )}
      <button
        type="button" disabled={submitting}
        onClick={() => onToggleConfirm("dismiss")}
        className={tertiaryBtn}
      >
        {flag.status === "hidden" ? "Restore photo & dismiss" : "Dismiss"}
      </button>
      {canConfirmHold(flag) && (
        <button
          type="button" disabled={submitting}
          onClick={() => onToggleConfirm("hold")}
          className={dangerBtn}
        >
          Hold payouts
        </button>
      )}
      {canHide(flag) && (
        <button
          type="button" disabled={submitting}
          onClick={onHide}
          className={dangerBtn}
        >
          Hide…
        </button>
      )}
    </>
  );
}

function FlagDetailBody({ flag }: { flag: Flag }) {
  const eventName = flagEventName(flag);

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
              loading="lazy"
              decoding="async"
              className="w-full h-full object-cover"
            />
          </div>
        ) : (
          <LogoPlaceholder size="lg" label={flag.photoSnapshot.alt} className="aspect-[16/9] rounded-2xl border border-line" />
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

      {isRelevanceFlag(flag) && (
        <>
          {flag.payoutHoldAt && (
            <p className="font-sans text-sm font-medium text-ink tnum">
              Payouts on hold since {formatLongDate(flag.payoutHoldAt)}
            </p>
          )}
          <FlagRelevanceEvidence flag={flag} />
        </>
      )}

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
