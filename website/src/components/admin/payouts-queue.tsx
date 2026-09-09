"use client";

import { useEffect, useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { AdminPayoutRow } from "@/components/admin/admin-payout-row";
import { AdminPayoutBulkBar } from "@/components/admin/admin-payout-bulk-bar";
import { AdminPayoutBulkApproveModal } from "@/components/admin/admin-payout-bulk-approve-modal";
import { AdminPayoutHoldModal } from "@/components/admin/admin-payout-hold-modal";
import { AdminPayoutMarkPaidModal } from "@/components/admin/admin-payout-mark-paid-modal";
import {
  AdminPayoutReasonModal,
  type PayoutReasonKind,
} from "@/components/admin/admin-payout-reason-modal";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { PayoutAccountCard } from "@/components/admin/payout-account-card";
import { ReadyToSendCard } from "@/components/admin/ready-to-send-card";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import {
  useAdminPayoutStore,
  mergePayoutsWithOverrides,
} from "@/store/admin-payout-store";
import { useAdminPayouts, usePayoutActions, EMPTY_PAYOUTS } from "@/hooks/use-admin-data";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import {
  useDrawerVerbs,
  useQueueKeyboardNav,
} from "@/hooks/use-admin-keyboard";
import {
  payoutMethodLabel,
  ADMIN_PAYOUT_STATUS_LABEL,
  type AdminPayoutCycle,
  type AdminPayoutStatus,
} from "@/lib/admin-payouts";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";

interface SingleHoldTarget {
  kind: "single";
  payoutId: string;
  amount: number;
}
interface BulkHoldTarget {
  kind: "bulk";
  payoutIds: string[];
  totalAmount: number;
}
type HoldTarget = SingleHoldTarget | BulkHoldTarget | null;

type MarkPaidTarget = AdminPayoutCycle;

// Terminal states carry no further admin action and can't be bulk-selected.
function isTerminal(status: AdminPayoutStatus): boolean {
  return (
    status === "paid" ||
    status === "failed" ||
    status === "returned" ||
    status === "rejected" ||
    status === "cancelled"
  );
}

function statusTone(status: AdminPayoutStatus): AdminStatusPillTone {
  switch (status) {
    case "pending_review":
      return "amber";
    case "approved":
      return "fresh";
    case "processing":
    case "held":
      return "ink";
    case "paid":
      return "slate";
    case "failed":
    case "returned":
      return "amber";
    case "rejected":
    case "cancelled":
      return "muted";
  }
}

// Payouts queue body — extracted from /admin/payouts/page.tsx so the
// universal /admin/inbox can render the same four slabs + bulk-select
// bar + three confirm modals under the chip filter.
//
// Selection state is local (Set<string>), never URL or Zustand. The
// floating bulk bar is `position: fixed` so it docks on whichever page
// hosts the queue. ESC clears selection when no modal/drawer is open.
//
// Phase 3 adds the focus-mode drawer. Row click is bulk-aware (selection
// empty → drawer; selection ≥ 1 → toggle checkbox); the chevron always
// opens the drawer. URL state via ?row=<cycleId>. Drawer footer shows the
// status-appropriate verbs; the existing Hold / Mark-paid / Bulk-approve
// modals (already at queue level) are shared between row-inline triggers,
// drawer-footer triggers, and the bulk bar — single source of truth.

export function PayoutsQueue() {
  const overrides = useAdminPayoutStore((s) => s.overrides);
  const { approve, hold, markPaid, reject, markProcessing, markFailed, markReturned, bulkApprove, bulkHold } = usePayoutActions();
  const { showToast } = useToast();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkApproveOpen, setBulkApproveOpen] = useState(false);
  const [holdTarget, setHoldTarget] = useState<HoldTarget>(null);
  const [markPaidTarget, setMarkPaidTarget] = useState<MarkPaidTarget | null>(
    null,
  );
  // reject / mark-failed / mark-returned all capture a reason via one modal.
  const [reasonTarget, setReasonTarget] = useState<{
    kind: PayoutReasonKind;
    cycle: AdminPayoutCycle;
  } | null>(null);
  const [rowId, setRowId] = useUrlState<string>("row", "");

  // Canonical server rows drive every status and available action.
  const serverPayouts = useAdminPayouts() ?? EMPTY_PAYOUTS;
  const effective = useMemo(
    () => mergePayoutsWithOverrides(serverPayouts, overrides),
    [serverPayouts, overrides],
  );

  const byId = useMemo(() => {
    const map = new Map<string, AdminPayoutCycle>();
    for (const c of effective) map.set(c.id, c);
    return map;
  }, [effective]);

  const openRow = useMemo<AdminPayoutCycle | null>(() => {
    if (!rowId) return null;
    return byId.get(rowId) ?? null;
  }, [rowId, byId]);

  useEffect(() => {
    if (rowId && !openRow) setRowId("");
  }, [rowId, openRow, setRowId]);

  useEffect(() => {
    setSelected((prev) => {
      let mutated = false;
      const next = new Set<string>();
      for (const id of prev) {
        const cycle = byId.get(id);
        if (cycle && !isTerminal(cycle.status)) next.add(id);
        else mutated = true;
      }
      return mutated ? next : prev;
    });
  }, [byId]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (
        bulkApproveOpen ||
        holdTarget !== null ||
        markPaidTarget !== null ||
        reasonTarget !== null ||
        openRow
      )
        return;
      if (selected.size === 0) return;
      setSelected(new Set());
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [
    selected,
    bulkApproveOpen,
    holdTarget,
    markPaidTarget,
    reasonTarget,
    openRow,
  ]);

  const pending = useMemo(
    () =>
      [...effective.filter((c) => c.status === "pending_review")].sort(
        (a, b) => a.submittedAt.localeCompare(b.submittedAt),
      ),
    [effective],
  );
  // approved + processing share a slab — both are "reviewed, payment in motion";
  // each row still shows its own status pill + status-appropriate actions.
  const inProgress = useMemo(
    () =>
      [
        ...effective.filter(
          (c) => c.status === "approved" || c.status === "processing",
        ),
      ].sort((a, b) => (b.reviewedAt ?? "").localeCompare(a.reviewedAt ?? "")),
    [effective],
  );
  const held = useMemo(
    () =>
      [...effective.filter((c) => c.status === "held")].sort((a, b) =>
        (b.reviewedAt ?? "").localeCompare(a.reviewedAt ?? ""),
      ),
    [effective],
  );
  const paid = useMemo(
    () =>
      [...effective.filter((c) => c.status === "paid")].sort((a, b) =>
        (b.paidAt ?? "").localeCompare(a.paidAt ?? ""),
      ),
    [effective],
  );
  // Terminal, non-paid states in one read-only slab so nothing disappears from
  // the admin's view (a failed/returned payout must stay visible).
  const closed = useMemo(
    () =>
      [
        ...effective.filter(
          (c) =>
            c.status === "failed" ||
            c.status === "returned" ||
            c.status === "rejected" ||
            c.status === "cancelled",
        ),
      ].sort((a, b) =>
        (b.reviewedAt ?? b.submittedAt).localeCompare(
          a.reviewedAt ?? a.submittedAt,
        ),
      ),
    [effective],
  );

  const selectedCycles = useMemo(
    () =>
      Array.from(selected)
        .map((id) => byId.get(id))
        .filter((c): c is AdminPayoutCycle => !!c),
    [selected, byId],
  );

  const totalSelected = selectedCycles.reduce((acc, c) => acc + c.amount, 0);
  const approveDisabled =
    selectedCycles.length === 0 ||
    selectedCycles.some((c) => c.status === "approved");
  const approveDisabledReason =
    selectedCycles.length === 0
      ? null
      : selectedCycles.some((c) => c.status === "approved")
        ? "Some selected payouts are already approved"
        : null;

  const selectionActive = selected.size > 0;
  const drawerEscDisabled =
    bulkApproveOpen ||
    holdTarget !== null ||
    markPaidTarget !== null ||
    reasonTarget !== null;

  // Pagination — load-more model. Pending review keeps full visibility
  // because the bulk-bar + J/K nav target that triage list. Read-only
  // history slabs start at 10 and grow by 10 per click.
  const [inProgressLoaded, setInProgressLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const [heldLoaded, setHeldLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const [paidLoaded, setPaidLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const [closedLoaded, setClosedLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const inProgressVisible = useMemo(
    () => inProgress.slice(0, inProgressLoaded),
    [inProgress, inProgressLoaded],
  );
  const heldVisible = useMemo(
    () => held.slice(0, heldLoaded),
    [held, heldLoaded],
  );
  const paidVisible = useMemo(
    () => paid.slice(0, paidLoaded),
    [paid, paidLoaded],
  );
  const closedVisible = useMemo(
    () => closed.slice(0, closedLoaded),
    [closed, closedLoaded],
  );

  // Keyboard nav iterates pending → approved → held → paid in render
  // order. Drawer verbs are status-conditional: E approves a pending or
  // held cycle, H opens the Hold reason modal whenever the cycle isn't
  // already on hold. Mark-paid stays click-only — its modal asks for a
  // payment reference string and isn't sensible to trigger from a
  // single keystroke. Use the visible slices for paginated slabs so J/K
  // only visits rows currently rendered.
  const rowIds = useMemo(
    () =>
      [
        ...pending,
        ...inProgressVisible,
        ...heldVisible,
        ...paidVisible,
        ...closedVisible,
      ].map((c) => c.id),
    [pending, inProgressVisible, heldVisible, paidVisible, closedVisible],
  );
  const queueNavDisabled = openRow !== null || drawerEscDisabled;
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: queueNavDisabled,
    onActivate: (id) => handleRowActivate(id),
  });
  useEffect(() => {
    if (rowId) setFocusedId(rowId);
  }, [rowId, setFocusedId]);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    setSelected(new Set());
  }

  function handleRowActivate(id: string) {
    setFocusedId(id);
    if (selectionActive) toggleSelect(id);
    else setRowId(id);
  }

  function handleOpenDrawer(id: string) {
    setFocusedId(id);
    setRowId(id);
  }

  async function handleSingleApprove(cycle: AdminPayoutCycle) {
    const ok = await approve(cycle.id);
    if (ok) showToast({ kind: "success", message: `Approval recorded · ${cycle.id}` });
  }
  function handleSingleHoldRequest(cycle: AdminPayoutCycle) {
    setHoldTarget({
      kind: "single",
      payoutId: cycle.id,
      amount: cycle.amount,
    });
  }
  function handleSingleMarkPaid(cycle: AdminPayoutCycle) {
    setMarkPaidTarget(cycle);
  }

  async function handleDrawerApprove() {
    if (!openRow) return;
    const id = openRow.id;
    const ok = await approve(id);
    if (ok) {
      showToast({ kind: "success", message: `Approval recorded · ${id}` });
      setRowId("");
    }
  }
  function handleDrawerHoldRequest() {
    if (!openRow) return;
    setHoldTarget({
      kind: "single",
      payoutId: openRow.id,
      amount: openRow.amount,
    });
  }
  function handleDrawerMarkPaidRequest() {
    if (!openRow) return;
    setMarkPaidTarget(openRow);
  }

  // mark-processing takes no reason — fire directly. The drawer stays open
  // (a processing cycle is still actionable: mark paid / mark failed).
  async function runMarkProcessing(id: string) {
    const ok = await markProcessing(id);
    if (ok) showToast({ kind: "success", message: `Processing · ${id}` });
  }
  function handleSingleMarkProcessing(cycle: AdminPayoutCycle) {
    void runMarkProcessing(cycle.id);
  }
  function handleDrawerMarkProcessing() {
    if (openRow) void runMarkProcessing(openRow.id);
  }

  // reject / mark-failed / mark-returned are drawer-only (they need a reason
  // and are less common than the inline triage actions).
  function handleDrawerReject() {
    if (openRow) setReasonTarget({ kind: "reject", cycle: openRow });
  }
  function handleDrawerMarkFailed() {
    if (openRow) setReasonTarget({ kind: "failed", cycle: openRow });
  }
  function handleDrawerMarkReturned() {
    if (openRow) setReasonTarget({ kind: "returned", cycle: openRow });
  }

  async function handleReasonSubmit(reason: string) {
    if (!reasonTarget) return;
    const { kind, cycle } = reasonTarget;
    setReasonTarget(null);
    const fn =
      kind === "reject" ? reject : kind === "failed" ? markFailed : markReturned;
    const ok = await fn(cycle.id, reason);
    if (ok) {
      const label =
        kind === "reject"
          ? "Rejected"
          : kind === "failed"
            ? "Marked failed"
            : "Marked returned";
      showToast({
        kind: kind === "reject" ? "info" : "success",
        message: `${label} · ${cycle.id}`,
      });
      // All three land in a terminal state — close the drawer if it was open.
      if (openRow && openRow.id === cycle.id) setRowId("");
    }
  }

  async function handleBulkApproveConfirm() {
    const ids = Array.from(selected);
    setBulkApproveOpen(false);
    clearSelection();
    const ok = await bulkApprove(ids);
    if (ok) {
      showToast({
        kind: "success",
        message: `Approvals recorded · ${ids.length} payouts`,
      });
    }
  }

  function handleBulkHoldRequest() {
    setHoldTarget({
      kind: "bulk",
      payoutIds: Array.from(selected),
      totalAmount: totalSelected,
    });
  }

  async function handleHoldSubmit(reason: string) {
    if (!holdTarget) return;
    const target = holdTarget;
    setHoldTarget(null);
    if (target.kind === "single") {
      const ok = await hold(target.payoutId, reason);
      if (ok) {
        showToast({
          kind: "info",
          message: `Held · ${target.payoutId}`,
        });
        // If the held cycle was the drawer's open row, close the drawer so
        // the queue state and the open detail don't drift apart.
        if (openRow && openRow.id === target.payoutId) setRowId("");
      }
    } else {
      clearSelection();
      const ok = await bulkHold(target.payoutIds, reason);
      if (ok) {
        showToast({
          kind: "info",
          message: `Held · ${target.payoutIds.length} payouts`,
        });
      }
    }
  }

  async function handleMarkPaidSubmit(reference: string) {
    if (!markPaidTarget) return;
    const target = markPaidTarget;
    setMarkPaidTarget(null);
    const ok = await markPaid(target.id, reference);
    if (ok) {
      showToast({
        kind: "success",
        message: `Paid · ${target.id}`,
      });
      if (openRow && openRow.id === target.id) setRowId("");
    }
  }

  useDrawerVerbs({
    active: openRow !== null && (openRow.provider ?? "manual") === "manual" && !drawerEscDisabled,
    verbs: openRow
      ? {
          E:
            openRow.status === "pending_review" || openRow.status === "held"
              ? handleDrawerApprove
              : undefined,
          H:
            openRow.status === "pending_review" ||
            openRow.status === "approved" ||
            openRow.status === "processing"
              ? handleDrawerHoldRequest
              : undefined,
        }
      : {},
  });

  return (
    <>
      <PayoutSlab
        id="pending"
        number="01"
        title="Pending review"
        caption="Awaiting first decision"
        totalCount={pending.length}
        rows={pending}
        empty="No payouts pending review."
        selected={selected}
        selectionActive={selectionActive}
        toggleSelect={toggleSelect}
        onActivate={handleRowActivate}
        onOpenDrawer={handleOpenDrawer}
        focusedId={focusedId}
        onApprove={handleSingleApprove}
        onHold={handleSingleHoldRequest}
        onMarkPaid={handleSingleMarkPaid}
        onMarkProcessing={handleSingleMarkProcessing}
      />
      <PayoutSlab
        id="in-progress"
        number="02"
        title="In progress"
        caption="Approved · payment in motion"
        totalCount={inProgress.length}
        rows={inProgressVisible}
        empty="No payouts in progress."
        selected={selected}
        selectionActive={selectionActive}
        toggleSelect={toggleSelect}
        onActivate={handleRowActivate}
        onOpenDrawer={handleOpenDrawer}
        focusedId={focusedId}
        onApprove={handleSingleApprove}
        onHold={handleSingleHoldRequest}
        onMarkPaid={handleSingleMarkPaid}
        onMarkProcessing={handleSingleMarkProcessing}
        loadMore={{
          shown: inProgressVisible.length,
          total: inProgress.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setInProgressLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <PayoutSlab
        id="held"
        number="03"
        title="Held"
        caption="Paused with reason"
        totalCount={held.length}
        rows={heldVisible}
        empty="No held payouts."
        selected={selected}
        selectionActive={selectionActive}
        toggleSelect={toggleSelect}
        onActivate={handleRowActivate}
        onOpenDrawer={handleOpenDrawer}
        focusedId={focusedId}
        onApprove={handleSingleApprove}
        onHold={handleSingleHoldRequest}
        onMarkPaid={handleSingleMarkPaid}
        onMarkProcessing={handleSingleMarkProcessing}
        loadMore={{
          shown: heldVisible.length,
          total: held.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setHeldLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <PayoutSlab
        id="paid"
        number="04"
        title="Paid"
        caption="Settled"
        totalCount={paid.length}
        rows={paidVisible}
        empty="No paid payouts yet."
        selected={selected}
        selectionActive={selectionActive}
        toggleSelect={toggleSelect}
        onActivate={handleRowActivate}
        onOpenDrawer={handleOpenDrawer}
        focusedId={focusedId}
        onApprove={handleSingleApprove}
        onHold={handleSingleHoldRequest}
        onMarkPaid={handleSingleMarkPaid}
        onMarkProcessing={handleSingleMarkProcessing}
        loadMore={{
          shown: paidVisible.length,
          total: paid.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setPaidLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <PayoutSlab
        id="closed"
        number="05"
        title="Closed"
        caption="Rejected, cancelled, failed or returned"
        totalCount={closed.length}
        rows={closedVisible}
        empty="No closed payouts."
        selected={selected}
        selectionActive={selectionActive}
        toggleSelect={toggleSelect}
        onActivate={handleRowActivate}
        onOpenDrawer={handleOpenDrawer}
        focusedId={focusedId}
        onApprove={handleSingleApprove}
        onHold={handleSingleHoldRequest}
        onMarkPaid={handleSingleMarkPaid}
        onMarkProcessing={handleSingleMarkProcessing}
        loadMore={{
          shown: closedVisible.length,
          total: closed.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setClosedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />

      <AdminPayoutBulkBar
        count={selected.size}
        totalAmount={totalSelected}
        approveDisabled={approveDisabled}
        approveDisabledReason={approveDisabledReason}
        onApprove={() => setBulkApproveOpen(true)}
        onHold={handleBulkHoldRequest}
        onClear={clearSelection}
      />

      <AdminPayoutBulkApproveModal
        open={bulkApproveOpen}
        onCancel={() => setBulkApproveOpen(false)}
        onConfirm={handleBulkApproveConfirm}
        count={selected.size}
        totalAmount={totalSelected}
      />
      {holdTarget && (
        <AdminPayoutHoldModal
          open={true}
          onClose={() => setHoldTarget(null)}
          onSubmit={handleHoldSubmit}
          payoutIds={
            holdTarget.kind === "single"
              ? [holdTarget.payoutId]
              : holdTarget.payoutIds
          }
          totalAmount={
            holdTarget.kind === "single"
              ? holdTarget.amount
              : holdTarget.totalAmount
          }
        />
      )}
      {markPaidTarget && (
        <AdminPayoutMarkPaidModal
          open={true}
          onClose={() => setMarkPaidTarget(null)}
          onSubmit={handleMarkPaidSubmit}
          cycle={markPaidTarget}
        />
      )}
      {reasonTarget && (
        <AdminPayoutReasonModal
          open={true}
          kind={reasonTarget.kind}
          payoutId={reasonTarget.cycle.id}
          amount={reasonTarget.cycle.amount}
          onClose={() => setReasonTarget(null)}
          onSubmit={handleReasonSubmit}
        />
      )}

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          escDisabled={drawerEscDisabled}
          kicker={`Payout · ${openRow.id}`}
          title={openRow.brandName ?? openRow.photographerName}
          rightHeader={
            <AdminStatusPill
              label={ADMIN_PAYOUT_STATUS_LABEL[openRow.status]}
              tone={statusTone(openRow.status)}
            />
          }
          subtitle={<PayoutSubtitle cycle={openRow} />}
          actions={
            <PayoutDrawerActions
              cycle={openRow}
              onApprove={handleDrawerApprove}
              onHold={handleDrawerHoldRequest}
              onMarkPaid={handleDrawerMarkPaidRequest}
              onMarkProcessing={handleDrawerMarkProcessing}
              onReject={handleDrawerReject}
              onMarkFailed={handleDrawerMarkFailed}
              onMarkReturned={handleDrawerMarkReturned}
            />
          }
        >
          <PayoutDetailBody cycle={openRow} />
        </AdminDetailDrawer>
      )}
    </>
  );
}

export function usePendingPayoutsCount(): number {
  const overrides = useAdminPayoutStore((s) => s.overrides);
  const serverPayouts = useAdminPayouts() ?? EMPTY_PAYOUTS;
  return useMemo(
    () =>
      mergePayoutsWithOverrides(serverPayouts, overrides).filter(
        (c) => c.status === "pending_review",
      ).length,
    [serverPayouts, overrides],
  );
}

export function usePendingPayoutsTotal(): number {
  const overrides = useAdminPayoutStore((s) => s.overrides);
  const serverPayouts = useAdminPayouts() ?? EMPTY_PAYOUTS;
  return useMemo(
    () =>
      mergePayoutsWithOverrides(serverPayouts, overrides)
        .filter((c) => c.status === "pending_review")
        .reduce((acc, c) => acc + c.amount, 0),
    [serverPayouts, overrides],
  );
}

function PayoutSlab({
  id,
  number,
  title,
  caption,
  rows,
  totalCount,
  empty,
  selected,
  selectionActive,
  focusedId,
  toggleSelect,
  onActivate,
  onOpenDrawer,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
  loadMore,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  rows: AdminPayoutCycle[];
  totalCount: number;
  empty: string;
  selected: Set<string>;
  selectionActive: boolean;
  focusedId: string | null;
  toggleSelect: (id: string) => void;
  onActivate: (id: string) => void;
  onOpenDrawer: (id: string) => void;
  onApprove: (cycle: AdminPayoutCycle) => void;
  onHold: (cycle: AdminPayoutCycle) => void;
  onMarkPaid: (cycle: AdminPayoutCycle) => void;
  onMarkProcessing: (cycle: AdminPayoutCycle) => void;
  loadMore?: {
    shown: number;
    total: number;
    increment: number;
    onLoadMore: () => void;
  };
}) {
  const noun = totalCount === 1 ? "payout" : "payouts";
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
          <ul className="space-y-4">
            {rows.map((row) => (
              <li key={row.id}>
                <AdminPayoutRow
                  cycle={row}
                  selected={selected.has(row.id)}
                  selectionActive={selectionActive}
                  focused={focusedId === row.id}
                  onToggleSelect={() => toggleSelect(row.id)}
                  onActivate={() => onActivate(row.id)}
                  onOpenDrawer={() => onOpenDrawer(row.id)}
                  onApprove={() => onApprove(row)}
                  onHold={() => onHold(row)}
                  onMarkPaid={() => onMarkPaid(row)}
                  onMarkProcessing={() => onMarkProcessing(row)}
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

// `submittedAt` is the photographer's request time (BE maps it from
// `cycle.createdAt`, the same source as the photographer-side `requestedAt`),
// so admin and photographer now read the same date for the same payout. The
// old `weekOf → +6d` range described scheduled weekly cycles, which the
// request-based flow replaced on 2026-05-19 — it was the last live consumer of
// the deprecated `weekOf`. No `dateOnly` flag: this is a full ISO timestamp.
function PayoutSubtitle({ cycle }: { cycle: AdminPayoutCycle }) {
  return (
    <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
      {cycle.handle ? `@${cycle.handle}` : "—"}
      <span className="text-slate-soft"> · </span>
      {formatLongDate(cycle.submittedAt)}
      <span className="text-slate-soft"> · </span>
      {formatPrice(cycle.amount)}
    </p>
  );
}

// Full per-status action matrix — mirrors AdminPayoutService.ensureFrom so a
// shown button never 409s:
//   pending_review → Hold, Reject, Approve      approved → Hold, Reject, Mark processing, Mark paid
//   processing     → Hold, Mark failed, Mark paid   held → Reject, Approve
//   paid           → Mark returned               failed/returned/rejected/cancelled → none
function PayoutDrawerActions({
  cycle,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
  onReject,
  onMarkFailed,
  onMarkReturned,
}: {
  cycle: AdminPayoutCycle;
  onApprove: () => void;
  onHold: () => void;
  onMarkPaid: () => void;
  onMarkProcessing: () => void;
  onReject: () => void;
  onMarkFailed: () => void;
  onMarkReturned: () => void;
}) {
  const PILL =
    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink border border-line hover:bg-ink hover:text-surface hover:border-ink transition-colors rounded-full px-5 py-2";
  const PRIMARY =
    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-surface bg-fresh hover:bg-fresh-deep transition-colors rounded-full px-5 py-2";
  const GHOST =
    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2";

  const s = cycle.status;

  if (cycle.provider && cycle.provider !== "manual") {
    return <p className="font-sans text-sm text-slate-soft">Payment provider manages this transfer.</p>;
  }
  if (s === "failed" || s === "returned" || s === "rejected" || s === "cancelled") {
    return (
      <p className="font-sans text-sm text-slate-soft">No further actions.</p>
    );
  }
  if (s === "paid") {
    return (
      <button type="button" onClick={onMarkReturned} className={PILL}>
        Mark returned…
      </button>
    );
  }

  return (
    <>
      {(s === "pending_review" || s === "approved" || s === "processing") && (
        <button type="button" onClick={onHold} className={GHOST}>
          Hold…
        </button>
      )}
      {(s === "pending_review" || s === "approved" || s === "held") && (
        <button type="button" onClick={onReject} className={PILL}>
          Reject…
        </button>
      )}
      {s === "approved" && (
        <button type="button" onClick={onMarkProcessing} className={PILL}>
          Mark processing
        </button>
      )}
      {s === "processing" && (
        <button type="button" onClick={onMarkFailed} className={PILL}>
          Mark failed…
        </button>
      )}
      {(s === "approved" || s === "processing") && (
        <button type="button" onClick={onMarkPaid} className={PILL}>
          Mark paid…
        </button>
      )}
      {(s === "pending_review" || s === "held") && (
        <button type="button" onClick={onApprove} className={PRIMARY}>
          Approve
        </button>
      )}
    </>
  );
}

function PayoutDetailBody({ cycle }: { cycle: AdminPayoutCycle }) {
  // "approved" is the post-review, pre-payment state — the moment the admin
  // actually has to send money. Surface the focal ReadyToSendCard at the
  // top; we hide the duplicate inline PayoutAccountCard below so the same
  // account info doesn't appear twice on the same screen.
  const isReadyToSend = cycle.status === "approved";

  return (
    <div className="space-y-10">
      {isReadyToSend && <ReadyToSendCard cycle={cycle} />}
      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Payout
        </Kicker>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow label="Payout id" value={cycle.id} mono />
          {/* No "Week of" row: payouts are request-based, so there is no week
              they cover. `Submitted` below is the real date and has been
              backed by `cycle.createdAt` all along (BE AdminPayoutService
              .hydrateOne). */}
          <FieldRow
            label="Submitted"
            value={formatLongDate(cycle.submittedAt)}
            mono
          />
          <FieldRow
            label="Sales"
            value={`${cycle.itemCount} ${cycle.itemCount === 1 ? "sale" : "sales"}`}
            mono
          />
          <FieldRow
            label="Amount"
            value={formatPrice(cycle.amount)}
            mono
          />
          <FieldRow
            label="Method"
            value={payoutMethodLabel(cycle.method)}
            mono={false}
          />
        </dl>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Photographer
        </Kicker>
        <div className="space-y-3">
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <p className="font-display text-lg text-ink">
              {cycle.brandName ?? cycle.photographerName}
            </p>
            {cycle.handle && (
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mt-1">
                @{cycle.handle}
              </p>
            )}
          </div>
          {!isReadyToSend && <PayoutAccountCard cycle={cycle} mode="drawer" />}
        </div>
      </section>

      {cycle.status === "held" && cycle.holdReason && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Hold reason
          </Kicker>
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <p className="font-sans text-sm text-ink-soft">
              {cycle.holdReason}
            </p>
            {cycle.reviewedAt && (
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-2 tnum">
                Held {formatLongDate(cycle.reviewedAt)}
              </p>
            )}
          </div>
        </section>
      )}

      {cycle.status === "paid" && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Payment
          </Kicker>
          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
            {cycle.paymentReference && (
              <FieldRow
                label="Reference"
                value={cycle.paymentReference}
                mono
              />
            )}
            {cycle.paidAt && (
              <FieldRow
                label="Paid at"
                value={formatLongDate(cycle.paidAt)}
                mono
              />
            )}
          </dl>
        </section>
      )}

      {cycle.status === "approved" && cycle.reviewedAt && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Approval
          </Kicker>
          <p className="font-sans text-sm text-ink-soft tnum">
            Approved {formatLongDate(cycle.reviewedAt)} — awaiting payment.
          </p>
          {cycle.secondApprovedBy && (
            <p className="font-sans text-sm text-slate-soft mt-1">
              Approved by two admins (2 of 2).
            </p>
          )}
        </section>
      )}

      {/* Two-person approval mid-flight: a large cycle stays pending_review
          after the first approval until a second, distinct admin signs off. */}
      {cycle.status === "pending_review" && cycle.approvedBy && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Approvals
          </Kicker>
          <p className="font-sans text-sm text-ink-soft">
            1 of 2 — first approval recorded. This payout needs a second, distinct
            admin before it can be sent.
          </p>
        </section>
      )}

      {cycle.status === "processing" && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Payment
          </Kicker>
          <p className="font-sans text-sm text-ink-soft">
            {(cycle.provider ?? "manual") === "manual"
              ? "Payment in transit — mark it paid once it lands, or mark it failed if it doesn't go through."
              : "Payment in transit. The payment provider will confirm its outcome."}
          </p>
        </section>
      )}

      {cycle.status === "rejected" && cycle.rejectedReason && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Rejected
          </Kicker>
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <p className="font-sans text-sm text-ink-soft">
              {cycle.rejectedReason}
            </p>
          </div>
        </section>
      )}

      {cycle.status === "failed" && cycle.failedReason && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Failed
          </Kicker>
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <p className="font-sans text-sm text-ink-soft">
              {cycle.failedReason}
            </p>
            <p className="font-sans text-sm text-slate-soft mt-2">
              The reserved funds returned to the photographer&apos;s balance to
              retry.
            </p>
          </div>
        </section>
      )}

      {cycle.status === "returned" && (
        <section>
          <Kicker as="p" tone="soft" className="mb-3">
            Returned
          </Kicker>
          <p className="font-sans text-sm text-ink-soft">
            The transfer left but bounced back. The photographer has been asked to
            check their payout account.
          </p>
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
