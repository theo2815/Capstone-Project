"use client";

import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from "react";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import { LoadError } from "@/components/ui/load-error";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SIZE,
  ROW_OPEN_BUTTON,
} from "@/components/ui/button-styles";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { KBD_CLASS } from "@/components/admin/keyboard-hint-pill";
import { AdminRejectModal } from "@/components/admin/admin-reject-modal";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { CollapsibleReviewSection } from "@/components/admin/admin-collapse";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { useAdminUserActions, useAdminUserDetail } from "@/hooks/use-admin-users";
import { AdminDecisionsTimeline } from "./admin-decisions-timeline";
import { useAdminUsers, EMPTY_ADMIN_USERS } from "@/hooks/use-admin-users";
import { useAuthStore } from "@/store/auth-store";
import { useToast } from "@/hooks/use-toast";
import { useUrlState } from "@/hooks/use-url-state";
import {
  useQueueKeyboardNav,
  useDrawerVerbs,
} from "@/hooks/use-admin-keyboard";
import {
  type AdminUserRow,
  type PhotographerSettingsSnapshot,
} from "@/lib/admin-user-registry";
import {
  completenessItems,
  computeCompleteness,
  photographerHref,
  snapshotFromSettings,
  useEffectivePhotographerSettings,
} from "@/lib/admin-photographer-view";
import {
  CompletenessSection,
  CoverSection,
  PayoutsSection,
  PublicUrlSection,
  SocialsSection,
  WatermarkSection,
} from "@/components/admin/photographer-review-sections";
import { formatMemberSince, sentenceCaseDate } from "@/lib/format";
import { formatRegionLabel } from "@/lib/ph-regions";
import { useRegions } from "@/hooks/use-regions";

// Verifications queue body — extracted from /admin/verifications/page.tsx so
// /admin/inbox can render the same three slabs (Open queue · Incomplete ·
// Recent decisions) under its own KPI-stripped header.
//
// Phase 2 added bulk-select on the Open queue: tick checkboxes to send
// back several pending photographers at once. Selection is
// local Set<string>, never URL or store. ESC clears the selection when
// no modal is open. The floating bar mirrors the payouts bar pattern.
//
// Phase 3 adds the focus-mode drawer. Click a pending row → drawer with
// the full review surface (photographer header, completeness panel,
// Reject in the footer). Profiles approve themselves once identity
// passes (Q7); only a profile held after a reset or reject gets an Approve. With at least one row selected, the row
// click toggles the checkbox instead of opening the drawer; the chevron
// on the right always opens. URL state via ?row=<userId> so refresh and
// browser-back behave; the drawer mirrors the chip route pattern.
//
// FT-41: the Verifications page puts Seller details in `afterIncomplete`, above the history; the Inbox passes nothing.
export function VerificationsQueue({ afterIncomplete }: { afterIncomplete?: ReactNode } = {}) {
  const users = useAdminUsers();
  const effective = users.data ?? EMPTY_ADMIN_USERS;
  const { isPending: loading, refetch } = users;
  const { reject, approveHeld, verifyBatch, isPending: actionPending } = useAdminUserActions();
  const { showToast } = useToast();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkRejectOpen, setBulkRejectOpen] = useState(false);
  const [drawerRejectOpen, setDrawerRejectOpen] = useState(false);
  const [approveOpen, setApproveOpen] = useState(false);
  const [rejectRevisions, setRejectRevisions] = useState<Record<string, number>>({});
  const [rowId, setRowId] = useUrlState<string>("row", "");
  const detail = useAdminUserDetail(rowId || null);
  const submitting = actionPending([...selected, ...(rowId ? [rowId] : [])]);

  const pending = useMemo(
    () =>
      effective.filter(
        (u) =>
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "pending" && u.suspendedAt === null,
      ),
    [effective],
  );

  const incomplete = useMemo(
    () =>
      effective.filter(
        (u) =>
          // Suspended photographers are left out, as in the directory.
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "incomplete" && u.suspendedAt === null,
      ),
    [effective],
  );

  // Lookup map for resolving log entries by userId without re-iterating.
  const byId = useMemo(() => {
    const map = new Map<string, AdminUserRow>();
    for (const row of effective) map.set(row.userId, row);
    return map;
  }, [effective]);

  // The row backing the open drawer. If the URL points at an id that has
  // left the pending pool (just approved/rejected, or stale link), we
  // close the drawer by clearing the URL state.
  const openRow = useMemo<AdminUserRow | null>(() => {
    if (!rowId) return null;
    const candidate = detail.isSuccess ? detail.data : byId.get(rowId);
    if (!candidate) return null;
    if (
      candidate.role !== "PHOTOGRAPHER" ||
      candidate.verificationStatus !== "pending" || candidate.suspendedAt !== null
    )
      return null;
    // The detail answer's adminHold first; an older backend's detail has none, so the list row's.
    return { ...candidate, adminHold: candidate.adminHold ?? byId.get(rowId)?.adminHold };
  }, [rowId, byId, detail.data, detail.isSuccess]);

  // C-05: the row the drawer last showed, and the id this admin just decided (so their own decision never reads as someone else's).
  const shownRowRef = useRef<AdminUserRow | null>(null);
  const ownDecisionRef = useRef<string | null>(null);
  useEffect(() => {
    if (openRow) shownRowRef.current = openRow;
  }, [openRow]);
  // Another row (or none): no decision of this admin's on it yet, and no approve confirm left open.
  useEffect(() => {
    ownDecisionRef.current = null;
    setApproveOpen(false);
  }, [rowId]);
  useEffect(() => {
    if (!openRow) setApproveOpen(false);
  }, [openRow]);
  useEffect(() => {
    if (rowId && detail.isSuccess && !detail.isFetching && !openRow && !submitting) {
      const shown = shownRowRef.current;
      shownRowRef.current = null;
      if (shown?.userId === rowId && ownDecisionRef.current !== rowId) {
        showToast({ kind: "info", message: `${shown.brandName ?? shown.name} was already decided by someone else.` });
      }
      setRowId("");
    }
  }, [rowId, openRow, setRowId, detail.isSuccess, detail.isFetching, submitting, showToast]);

  // Drop ids that have left the pending pool (e.g. just approved /
  // rejected — the row no longer renders a checkbox).
  const pendingIds = useMemo(
    () => new Set(pending.map((p) => p.userId)),
    [pending],
  );
  useEffect(() => {
    if (submitting) return;
    setSelected((prev) => {
      let mutated = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (pendingIds.has(id)) next.add(id);
        else mutated = true;
      }
      return mutated ? next : prev;
    });
  }, [pendingIds, submitting]);

  // ESC clears selection when no modal/drawer is open.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (submitting || bulkRejectOpen || drawerRejectOpen || openRow) return;
      if (selected.size === 0) return;
      setSelected(new Set());
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selected, bulkRejectOpen, drawerRejectOpen, openRow, submitting]);

  const selectionActive = selected.size > 0;
  const allSelected = pending.length > 0 && pending.every((p) => selected.has(p.userId));

  // Keyboard nav covers the Open queue only — the Incomplete and Recent
  // decisions slabs are read-only listings, not openable rows.
  const rowIds = useMemo(() => pending.map((p) => p.userId), [pending]);

  // Pagination — load-more model. Open queue stays full visibility so
  // J/K, the bulk-bar, and the drawer flow all see every pending row at
  // once. Read-only slabs (Incomplete + Recent decisions) start at 10
  // and grow by 10 per click.
  const [incompleteLoaded, setIncompleteLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const incompleteVisible = useMemo(
    () => incomplete.slice(0, incompleteLoaded),
    [incomplete, incompleteLoaded],
  );
  const queueNavDisabled =
    submitting || openRow !== null || bulkRejectOpen || drawerRejectOpen || approveOpen;
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: queueNavDisabled,
    onActivate: (id) => handleRowActivate(id),
  });

  // Mirror URL-driven and click-driven row openings into focusedId so
  // J/K resumes from the row the user just looked at.
  useEffect(() => {
    if (rowId) setFocusedId(rowId);
  }, [rowId, setFocusedId]);

  function toggleSelect(id: string) {
    if (submitting) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    if (submitting) return;
    setSelected(new Set());
  }

  function handleRowActivate(id: string) {
    if (submitting) return;
    setFocusedId(id);
    if (selectionActive) toggleSelect(id);
    else setRowId(id);
  }

  function handleOpenDrawer(id: string) {
    if (submitting) return;
    setFocusedId(id);
    setRowId(id);
  }

  function openRejection(bulk: boolean) {
    const rows = bulk ? Array.from(selected).map((id) => byId.get(id)!) : openRow ? [openRow] : [];
    setRejectRevisions(Object.fromEntries(rows.map((row) => [row.userId, row.verificationRevision])));
    if (bulk) setBulkRejectOpen(true);
    else setDrawerRejectOpen(true);
  }

  // A row ticked after the send-back form opened has no captured revision; the server would skip its stale check.
  const sendBackIds = Array.from(selected).filter((id) => pendingIds.has(id) && id in rejectRevisions);

  // Sends back in parallel so one BE failure doesn't block the rest. The
  // summary toast tallies successes vs failures so admin sees an honest
  // count — individual failures already surfaced their own error toast.
  async function handleBulkRejectSubmit(reason: string) {
    const sessionVersion = useAuthStore.getState().sessionVersion;
    const ids = sendBackIds;
    if (!ids.length) {
      showToast({ kind: "error", message: "These rows changed after this form opened. Close the form and tick them again." });
      return;
    }
    const results = await verifyBatch(ids.map((id) => ({ userId: id, verificationRevision: rejectRevisions[id] })), reason);
    if (!results) return false;
    if (useAuthStore.getState().sessionVersion !== sessionVersion) return;
    const failed = ids.filter((_, i) => !results[i]);
    setSelected(new Set(failed));
    const count = ids.length - failed.length;
    if (!failed.length) setBulkRejectOpen(false);
    const failedNames = failed.map((id) => { const row = byId.get(id); return row ? row.brandName ?? row.name : id; }).join(", ");
    if (count) showToast({
      kind: failed.length ? "error" : "success",
      message: failed.length ? `${count} of ${ids.length} sent back. Not sent back, still ticked: ${failedNames}.` : `Sent back · ${count} ${count === 1 ? "photographer" : "photographers"}`,
    });
    return !failed.length;
  }

  async function handleDrawerRejectSubmit(reason: string) {
    if (!openRow) return;
    const row = openRow;
      ownDecisionRef.current = row.userId;
      const ok = await reject(row.userId, reason, rejectRevisions[row.userId]);
      if (ok) {
        showToast({ kind: "success", message: `Sent back · ${row.brandName ?? row.name}` });
        setDrawerRejectOpen(false); setRowId("");
      }
      return ok;
  }

  async function handleApproveHeld() {
    if (!openRow) return;
    const row = openRow;
    ownDecisionRef.current = row.userId;
    if (await approveHeld(row.userId, row.verificationRevision)) {
      setApproveOpen(false);
      showToast({ kind: "success", message: `Hold released · ${row.brandName ?? row.name}` });
    }
  }

  // Drawer verb: R opens the reject reason modal. Disabled while the
  // modal is on top so a stray R doesn't punch through.
  useDrawerVerbs({
    active: openRow !== null && !drawerRejectOpen && !approveOpen && !submitting,
    verbs: openRow
      ? {
          R: () => openRejection(false),
        }
      : {},
  });

  // First-load indicator: only surface the spinner/error when we have no
  // cached rows to show. After the initial fetch lands, the SWR-style cache
  // keeps the queue visible while background revalidations run.
  const showFirstLoad = loading && effective.length === 0;
  const showError = users.isError && effective.length === 0;
  // No list yet: show no counts and no "nothing here" copy, which would read as a real zero.
  const noList = showFirstLoad || showError;

  return (
    <>
      {rowId && detail.isPending && !openRow && <div aria-busy="true"><Skeleton className="h-28 w-full rounded-2xl" /></div>}
      {rowId && detail.isError && <LoadError what="verification" onRetry={() => void detail.refetch()} className="text-sm text-error" />}
      {showFirstLoad && (
        <div className="space-y-4" aria-busy="true">
          {Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-28 w-full rounded-2xl" />)}
        </div>
      )}
      {showError && <LoadError what="the verifications" onRetry={() => void refetch()} className="text-sm text-error" />}
      <Slab
        titleStyle="heading"
        id="open-queue"
        number="01"
        title="Open queue"
        caption="Waiting for the ID check"
        trailing={noList ? undefined : `${pending.length} waiting`}
      >
        {pending.length === 0 ? (
          !noList && <QueueEmpty />
        ) : (
          <>
          <label className="mb-2 inline-flex min-h-11 cursor-pointer items-center gap-3 font-sans text-sm text-ink">
            <input
              type="checkbox"
              disabled={submitting}
              checked={allSelected}
              onChange={() => (allSelected ? clearSelection() : setSelected(new Set(rowIds)))}
              className="size-4 accent-ink cursor-pointer disabled:cursor-not-allowed"
            />
            Select all
          </label>
          <ul className="space-y-4">
            {pending.map((row) => (
              <li key={row.userId}>
                <PendingRow
                  row={row}
                  disabled={submitting}
                  selected={selected.has(row.userId)}
                  selectionActive={selectionActive}
                  focused={focusedId === row.userId}
                  onToggleSelect={() => toggleSelect(row.userId)}
                  onActivate={() => handleRowActivate(row.userId)}
                  onOpenDrawer={() => handleOpenDrawer(row.userId)}
                />
              </li>
            ))}
          </ul>
          </>
        )}
      </Slab>

      <Slab
        titleStyle="heading"
        id="incomplete"
        number="02"
        title="Incomplete"
        caption="Still filling in their settings. Nothing to do here."
        trailing={noList ? undefined : `${incomplete.length} not finished`}
        collapsed={incomplete.length === 0}
      >
        {incomplete.length === 0 ? (
          !noList && <p className="font-sans text-sm text-slate-soft">
            No photographers are part-way through their settings.
          </p>
        ) : (
          <>
            <ul className="space-y-3">
              {incompleteVisible.map((row) => (
                <li
                  key={row.userId}
                  className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-line pb-3"
                >
                  <div className="min-w-0">
                    <p className="font-display text-base text-ink truncate">
                      <Link
                        href={photographerHref(row)}
                        className="underline decoration-line underline-offset-4 hover:decoration-ink"
                      >
                        {row.brandName ?? row.name}
                      </Link>
                    </p>
                    <p className="font-sans text-sm text-slate-soft mt-1">
                      {row.handle ? `@${row.handle}` : "No handle yet"}
                      <span className="text-slate-soft"> · </span>
                      <CompletenessFraction snapshot={row.settingsSnapshot} />
                    </p>
                  </div>
                  <Kicker as="span" tone="soft" className="shrink-0">
                    They finish this themselves
                  </Kicker>
                </li>
              ))}
            </ul>
            <LoadMoreButton
              shown={incompleteVisible.length}
              total={incomplete.length}
              increment={PAGE_SIZE.ADMIN_INCREMENT}
              onLoadMore={() =>
                setIncompleteLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT)
              }
            />
          </>
        )}
      </Slab>

      {afterIncomplete}

      <Slab titleStyle="heading" id="recent-decisions" number={afterIncomplete ? "04" : "03"} title="Decision history">
        <AdminDecisionsTimeline />
      </Slab>

      {/* Room at the end, so the fixed bulk bar never covers what is last on the page (as on Payouts). */}
      {selectionActive && <div aria-hidden="true" className="h-36" />}
      <VerificationsBulkBar
        submitting={submitting}
        count={selected.size}
        onReject={() => openRejection(true)}
        onClear={clearSelection}
      />

      <AdminRejectModal

        submitting={submitting}
        open={bulkRejectOpen}
        onClose={() => setBulkRejectOpen(false)}
        onSubmit={handleBulkRejectSubmit}
        photographerName={`${sendBackIds.length} ${
          sendBackIds.length === 1 ? "photographer" : "photographers"
        }`}
        count={sendBackIds.length}
      />

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => { if (!submitting) setRowId(""); }}
          escDisabled={drawerRejectOpen || approveOpen || submitting}
          kicker="Verification"
          title={openRow.brandName ?? openRow.name}
          rightHeader={
            <span className={`inline-flex items-center font-mono uppercase ${KICKER_SIZE_CLASS.sm} rounded-full border border-line text-slate px-3 py-0.5`}>
              Pending
            </span>
          }
          subtitle={
            <p className="font-sans text-sm text-slate tnum break-words">
              {openRow.handle ? `@${openRow.handle}` : "No handle yet"}
              {(openRow.region || openRow.city) && (
                <>
                  <span className="text-slate-soft"> · </span>
                  {openRow.region || openRow.city}
                </>
              )}
              <span className="text-slate-soft"> · </span>
              <span className="tnum">
                Joined {sentenceCaseDate(formatMemberSince(openRow.createdAt))}
              </span>
            </p>
          }
          actions={
            <>
              <button
                type="button"
                disabled={submitting}
                aria-busy={submitting}
                onClick={() => openRejection(false)}
                className={`${BTN_DANGER} ${BTN_SIZE.sm}`}
              >
                Send back…<kbd aria-hidden="true" className={`${KBD_CLASS} hidden md:inline`}>R</kbd>
              </button>
              {openRow.adminHold && (
                <button
                  type="button"
                  disabled={submitting}
                  aria-busy={submitting}
                  onClick={() => setApproveOpen(true)}
                  className={`${BTN_PRIMARY} ${BTN_SIZE.sm}`}
                >
                  Approve…
                </button>
              )}
            </>
          }
        >
          {openRow.adminHold && (
            <p className="font-sans text-sm text-ink-soft mb-4">
              Held: it was sent back or reset before, so it needs your approval.
            </p>
          )}
          <VerificationDetailBody row={openRow} />
        </AdminDetailDrawer>
      )}

      {openRow && (
        <AdminRejectModal
          submitting={submitting}
          open={drawerRejectOpen}
          onClose={() => setDrawerRejectOpen(false)}
          onSubmit={handleDrawerRejectSubmit}
          photographerName={openRow.brandName ?? openRow.name}
        />
      )}

      {openRow && (
        <AdminConfirmModal
          open={approveOpen}
          submitting={submitting}
          pendingLabel="Approving…"
          onClose={() => setApproveOpen(false)}
          onConfirm={() => void handleApproveHeld()}
          title={`Approve ${openRow.brandName ?? openRow.name}?`}
          confirmLabel="Approve"
        >
          <p className="font-sans text-sm text-ink-soft">
            This releases the hold. If their ID check has passed, the profile goes live now and they are told. If it has not, the profile goes live by itself when the check passes.
          </p>
        </AdminConfirmModal>
      )}
    </>
  );
}

function PendingRow({
  row,
  disabled,
  selected,
  selectionActive,
  focused,
  onToggleSelect,
  onActivate,
  onOpenDrawer,
}: {
  row: AdminUserRow;
  disabled: boolean;
  selected: boolean;
  selectionActive: boolean;
  focused: boolean;
  onToggleSelect: () => void;
  onActivate: () => void;
  onOpenDrawer: () => void;
}) {
  const completeness = computeCompleteness(row.settingsSnapshot);
  // C-08: the dots are colour only, so the missing items are also named in text.
  const missing = row.settingsSnapshot
    ? completenessItems(row.settingsSnapshot).filter((item) => !item.filled).map((item) => item.label.toLowerCase())
    : [];
  const title = row.brandName ?? row.name;
  const hintId = useId();

  // Row pattern from AX-4: the title button covers the row (ROW_OPEN_BUTTON); inner controls sit above it.
  return (
    <article
      aria-busy={disabled}
      data-row-id={row.userId}
      className={`group relative rounded-2xl border bg-bone p-5 md:p-6 transition-colors cursor-pointer ${
        selected ? "border-ink" : "border-line hover:border-ink"
      } ${focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""}`}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        {/* The label gives the 16 px checkbox a 44 px hit area; the negative margin keeps the row's layout. */}
        <label className="relative z-10 -mx-3.5 -my-1.5 flex size-11 shrink-0 cursor-pointer items-center justify-center">
          <input
            type="checkbox"
            disabled={disabled}
            checked={selected}
            onChange={onToggleSelect}
            aria-label={`Select ${title}`}
            className="size-4 accent-ink cursor-pointer disabled:cursor-not-allowed"
          />
        </label>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-3 flex-wrap">
                {/* Proposal 11: name leads; details are sentence case, mono only for the handle and numbers. */}
                <h3 className="font-display text-xl font-semibold text-ink truncate">
                  <button type="button" onClick={onActivate} aria-describedby={hintId} className={ROW_OPEN_BUTTON}>
                    {title}
                  </button>
                </h3>
                <span id={hintId} className="sr-only">{selectionActive ? "Toggles selection" : "Opens the review"}</span>
                <p className="font-mono text-[13px] text-slate">
                  @{row.handle ?? "—"}
                </p>
                {row.adminHold && <AdminStatusPill label="Held" tone="amber" />}
              </div>
              <div className="mt-1 flex items-center gap-x-2 gap-y-1 flex-wrap font-sans text-sm text-slate">
                {(row.region || row.city) && <span>{row.region || row.city} ·</span>}
                <span>joined {sentenceCaseDate(formatMemberSince(row.createdAt))} ·</span>
                <CompletenessStrip snapshot={row.settingsSnapshot} />
                <span className="tnum">
                  {completeness.filled} of {completeness.total} fields
                </span>
                {missing.length > 0 && <span>· Missing: {missing.join(", ")}</span>}
                {row.settingsSnapshot && (
                  <span>
                    · <span className="tnum">{row.settingsSnapshot.socialCount}</span>{" "}
                    {row.settingsSnapshot.socialCount === 1 ? "social link" : "social links"} ·{" "}
                    <span className="tnum">{row.settingsSnapshot.payoutCount}</span>{" "}
                    {row.settingsSnapshot.payoutCount === 1 ? "payout account" : "payout accounts"}
                  </span>
                )}
              </div>
              <p className="font-sans text-sm text-ink-soft mt-2">
                Waits for the identity check. Decide it under{" "}
                <Link className="relative z-10 underline" href={`${ROUTES.ADMIN_VERIFICATIONS}#seller-details`}>Seller details</Link>.
              </p>
            </div>

            <button
              type="button"
              disabled={disabled}
              onClick={onOpenDrawer}
              aria-label={`Open review for ${title}`}
              className="relative z-10 disabled:opacity-50 disabled:cursor-not-allowed shrink-0 size-11 rounded-full text-slate hover:text-ink hover:bg-bone-deep transition-colors flex items-center justify-center"
            >
              <span aria-hidden="true" className="text-lg leading-none">
                ›
              </span>
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function VerificationDetailBody({ row }: { row: AdminUserRow }) {
  const settings = useEffectivePhotographerSettings(row.userId);
  const effective = settings.data;
  const { regions } = useRegions();

  const liveSnapshot = effective ? snapshotFromSettings(effective) : row.settingsSnapshot;

  const completeness = computeCompleteness(liveSnapshot);
  const resolvedRegion = effective?.region
    ? formatRegionLabel(
        regions,
        effective.region.regionCode,
        effective.region.provinceCode,
      )
    : null;
  const regionLine = resolvedRegion ?? row.region ?? "Not set";
  const publicHandle = effective?.handle?.trim() || row.handle || "";

  const socialsCount = effective?.socials.length ?? 0;
  const payoutsCount = effective?.payouts.length ?? 0;
  const avatarUrl = effective?.avatarUrl ?? row.avatarUrl ?? null;

  if (settings.isPending) return <div aria-busy="true" className="space-y-3">{Array.from({ length: 2 }, (_, i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div>;
  if (settings.isError) return <LoadError what="verification details" onRetry={() => void settings.refetch()} className="text-sm text-error" />;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 pb-2">
        <AvatarDisc
          name={row.brandName ?? row.name}
          size="md"
          tone="ink"
          avatarOverride={avatarUrl ? { dataUrl: avatarUrl } : null}
        />
        <div className="min-w-0">
          <p className="font-display text-lg md:text-xl text-ink break-words">
            {row.brandName ?? row.name}
          </p>
          <p className="font-mono text-[13px] text-slate-soft break-all">
            {row.email}
          </p>
        </div>
      </div>

      <CollapsibleReviewSection kicker="Who they are">
        <p className="font-sans text-sm text-ink-soft mb-4">
          ID photos are checked under{" "}
          <Link className="underline" href={`${ROUTES.ADMIN_VERIFICATIONS}#seller-details`}>Seller details</Link>.
        </p>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow label="Email" value={row.email} mono={false} />
          <FieldRow label="Legal name" value={row.name} mono={false} />
          <FieldRow
            label="Brand name"
            value={row.brandName ?? "—"}
            mono={false}
          />
          <FieldRow
            label="Handle"
            value={publicHandle ? `@${publicHandle}` : "—"}
            mono
          />
          <FieldRow label="City" value={row.city} mono={false} />
          <FieldRow label="Region" value={regionLine} mono={false} />
        </dl>
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Public profile cover">
        <CoverSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Public URL">
        <PublicUrlSection handle={publicHandle} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Watermark">
        <WatermarkSection
          effective={effective}
          brand={row.brandName ?? row.name}
        />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection
        kicker={
          <>
            Social &amp; verification links ·{" "}
            <span className="tnum">{socialsCount}</span>{" "}
            {socialsCount === 1 ? "link" : "links"}
          </>
        }
      >
        <SocialsSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection
        kicker={
          <>
            Payout accounts ·{" "}
            <span className="tnum">{payoutsCount}</span>{" "}
            {payoutsCount === 1 ? "account" : "accounts"}
          </>
        }
      >
        <PayoutsSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection
        kicker={
          <>
            Settings completeness ·{" "}
            <span className="tnum">{completeness.filled}/{completeness.total}</span> fields
          </>
        }
      >
        <CompletenessSection snapshot={liveSnapshot} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Account">
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow
            label="Joined"
            value={sentenceCaseDate(formatMemberSince(row.createdAt))}
            mono
          />
          <FieldRow label="Status" value="Pending review" mono={false} />
        </dl>
      </CollapsibleReviewSection>
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
      <Kicker as="dt" tone="soft">
        {label}
      </Kicker>
      <dd
        className={`mt-1 break-words ${
          mono
            ? "font-mono text-[13px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

function VerificationsBulkBar({
  count,
  submitting,
  onReject,
  onClear,
}: {
  count: number;
  submitting: boolean;
  onReject: () => void;
  onClear: () => void;
}) {
  if (count === 0) return null;

  return (
    <div
      role="region"
      aria-label="Verifications bulk action bar"
      className="fixed inset-x-4 bottom-4 lg:left-auto lg:right-8 lg:bottom-6 lg:max-w-xl z-40 rounded-2xl border border-ink bg-bone shadow-2xl px-4 md:px-6 py-4 animate-[fade-up_0.2s_ease-out_both]"
    >
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <p className="kicker">
          <span className="tnum">{count}</span> selected
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            disabled={submitting}
                  aria-busy={submitting}
            onClick={onReject}
            className={cn(BTN_DANGER, BTN_SIZE.sm)}
          >
            Send back {count}…
          </button>
          <button
            type="button"
            disabled={submitting}
                  aria-busy={submitting}
            onClick={onClear}
            className={cn(BTN_GHOST, BTN_SIZE.sm)}
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

function CompletenessStrip({
  snapshot,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
}) {
  if (!snapshot) {
    return null;
  }
  return (
    <div
      className="flex items-center gap-1.5"
      aria-label="Settings completeness"
    >
      {completenessItems(snapshot).map((item) => (
        <span
          key={item.label}
          title={`${item.label}${item.filled ? " ✓" : " — missing"}`}
          className={`size-2 rounded-full ${item.filled ? "bg-fresh" : "bg-line"}`}
          aria-hidden="true"
        />
      ))}
    </div>
  );
}

function CompletenessFraction({
  snapshot,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
}) {
  const c = computeCompleteness(snapshot);
  return (
    <span className="tnum">
      {c.filled}/{c.total} fields
    </span>
  );
}

function QueueEmpty() {
  return (
    <div className="rounded-2xl border border-dashed border-line p-10 text-center">
      <p className="font-display text-xl text-ink">No profiles waiting. You&apos;re caught up.</p>
    </div>
  );
}
