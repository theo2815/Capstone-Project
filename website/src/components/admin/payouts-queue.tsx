"use client";

import { useEffect, useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { AdminPayoutRow, PayoutFlagLine, PayoutTableHeader, payoutStatusTone } from "@/components/admin/admin-payout-row";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
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
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { useSearchParams } from "next/navigation";
import { useAdminPayoutById, useAdminPayoutCounts, useAdminPayoutsPage, usePayoutActions } from "@/hooks/use-admin-payouts";
import { BTN_DANGER, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import {
  useDrawerVerbs,
  useQueueKeyboardNav,
} from "@/hooks/use-admin-keyboard";
import {
  payoutMethodLabel,
  payoutActionsBlockedByHold,
  ADMIN_PAYOUT_STATUS_LABEL,
  type AdminPayoutCycle,
  type AdminPayoutStatus,
} from "@/lib/admin-payouts";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";

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

// Payouts queue body — extracted from /admin/payouts/page.tsx so the
// universal /admin/inbox can render the same status tabs (one slab for the
// chosen tab) + bulk-select bar + confirm modals under the chip filter.
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
  const { approve, hold, markPaid, reject, markProcessing, markFailed, markReturned, bulkApprove, bulkHold, isPending, pendingAction } = usePayoutActions();
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
  // A single Approve or Mark processing waits for this confirm, like bulk Approve.
  const [confirmTarget, setConfirmTarget] = useState<{
    kind: "approve" | "processing";
    cycle: AdminPayoutCycle;
  } | null>(null);
  const [rowId, setRowId] = useUrlState<string>("row", "");
  // Proposal 10: one status at a time, as tabs, instead of five stacked sections.
  const [tabRaw, setTab] = useUrlState<string>("status", "pending");
  const tab: PayoutTab = (PAYOUT_TABS as readonly string[]).includes(tabRaw) ? (tabRaw as PayoutTab) : "pending";

  const actionIds = holdTarget?.kind === "single" ? [holdTarget.payoutId]
    : holdTarget?.kind === "bulk" ? holdTarget.payoutIds
    : markPaidTarget ? [markPaidTarget.id]
    : reasonTarget ? [reasonTarget.cycle.id]
    : confirmTarget ? [confirmTarget.cycle.id]
    : bulkApproveOpen ? [...selected] : rowId ? [rowId] : [...selected];
  const submitting = isPending(actionIds);

  // AX-30 (PAY1): each tab is its own server list, paged, so no row drops off past 200.
  const [search, setSearch] = useUrlState<string>("q", "", { debounceMs: 300 });
  // The server search follows the URL, which trails typing by the 300 ms debounce.
  const q = (useSearchParams().get("q") ?? "").trim();
  const list = useAdminPayoutsPage(TAB_STATUSES[tab], q);
  const counts = useAdminPayoutCounts(q);
  const effective = list.items;

  const byId = useMemo(() => {
    const map = new Map<string, AdminPayoutCycle>();
    for (const c of effective) map.set(c.id, c);
    return map;
  }, [effective]);

  // A link may name a payout on another tab or page, so the drawer also reads it by id.
  const rowLookup = useAdminPayoutById(rowId);
  const openRow = rowId ? byId.get(rowId) ?? rowLookup.data ?? null : null;

  useEffect(() => {
    if (rowId && !openRow && rowLookup.isSuccess && !rowLookup.isFetching) setRowId("");
  }, [rowLookup.isSuccess, rowLookup.isFetching, rowId, openRow, setRowId]);

  // A ticked row whose status left the shown tab (a refetch, or a tab change by
  // button, Back/Forward or a ?status= link) drops out, so bulk actions never reach off-screen rows.
  useEffect(() => {
    setSelected((prev) => {
      let mutated = false;
      const next = new Set<string>();
      for (const id of prev) {
        const cycle = byId.get(id);
        if (cycle && PAYOUT_TAB_OF[cycle.status] === tab) next.add(id);
        else mutated = true;
      }
      return mutated ? next : prev;
    });
  }, [byId, tab]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (
        submitting ||
        bulkApproveOpen ||
        holdTarget !== null ||
        markPaidTarget !== null ||
        reasonTarget !== null ||
        confirmTarget !== null ||
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
    submitting,
    bulkApproveOpen,
    holdTarget,
    markPaidTarget,
    reasonTarget,
    confirmTarget,
    openRow,
  ]);

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
    selectedCycles.some((c) => c.status === "approved" || payoutActionsBlockedByHold(c));
  const approveDisabledReason =
    selectedCycles.length === 0
      ? null
      : selectedCycles.some((c) => c.status === "approved")
        ? "Some selected payouts are already approved"
        : selectedCycles.some(payoutActionsBlockedByHold)
          ? "Some selected payouts are on hold: open relevance flag"
          : null;

  const selectionActive = selected.size > 0;
  const drawerEscDisabled =
    submitting ||
    bulkApproveOpen ||
    holdTarget !== null ||
    markPaidTarget !== null ||
    reasonTarget !== null ||
    confirmTarget !== null;

  // Keyboard nav visits the loaded rows of the shown tab, in server order.
  // Drawer verbs are status-conditional: E opens the Approve confirm for a
  // pending or held cycle, H opens the Hold reason modal whenever the cycle isn't
  // already on hold. Mark-paid stays click-only — its modal asks for a
  // payment reference string and isn't sensible to trigger from a
  // single keystroke.
  const rowIds = useMemo(() => effective.map((c) => c.id), [effective]);
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
    if (isPending(actionIds)) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    if (isPending(actionIds)) return;
    setSelected(new Set());
  }

  function handleRowActivate(id: string) {
    if (isPending(actionIds)) return;
    setFocusedId(id);
    if (selectionActive) toggleSelect(id);
    else setRowId(id);
  }

  function handleOpenDrawer(id: string) {
    setFocusedId(id);
    setRowId(id);
  }

  function handleSingleApprove(cycle: AdminPayoutCycle) {
    setConfirmTarget({ kind: "approve", cycle });
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

  function handleDrawerApprove() {
    if (openRow) handleSingleApprove(openRow);
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

  function handleSingleMarkProcessing(cycle: AdminPayoutCycle) {
    setConfirmTarget({ kind: "processing", cycle });
  }
  function handleDrawerMarkProcessing() {
    if (openRow) handleSingleMarkProcessing(openRow);
  }

  // The confirm stays open with the pending label until the backend answers;
  // a failure keeps it open so the admin can try again.
  async function handleConfirmSubmit() {
    if (!confirmTarget) return;
    const { kind, cycle } = confirmTarget;
    const ok = await (kind === "approve" ? approve : markProcessing)(cycle.id);
    if (!ok) return;
    setConfirmTarget(null);
    showToast({ kind: "success", message: `${kind === "approve" ? "Approval recorded" : "Processing"} · ${cycle.id}` });
    // An approval closes its drawer; a processing payout stays actionable (mark paid / mark failed).
    if (kind === "approve" && openRow?.id === cycle.id) setRowId("");
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
    const fn =
      kind === "reject" ? reject : kind === "failed" ? markFailed : markReturned;
    const ok = await fn(cycle.id, reason);
    if (ok) {
      setReasonTarget(null);
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
    const ok = await bulkApprove(ids, (failedIds) => setSelected(new Set(failedIds)));
    if (ok) {
      setBulkApproveOpen(false);
      clearSelection();
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
    if (target.kind === "single") {
      const ok = await hold(target.payoutId, reason);
      if (ok) {
        setHoldTarget(null);
        showToast({
          kind: "info",
          message: `Held · ${target.payoutId}`,
        });
        // If the held cycle was the drawer's open row, close the drawer so
        // the queue state and the open detail don't drift apart.
        if (openRow && openRow.id === target.payoutId) setRowId("");
      }
      return ok;
    } else {
      const ok = await bulkHold(target.payoutIds, reason, (failedIds) => {
        setSelected(new Set(failedIds));
        setHoldTarget({
          kind: "bulk", payoutIds: failedIds,
          totalAmount: failedIds.reduce((total, id) => total + (byId.get(id)?.amount ?? 0), 0),
        });
      });
      if (ok) {
        setHoldTarget(null);
        clearSelection();
        showToast({
          kind: "info",
          message: `Held · ${target.payoutIds.length} payouts`,
        });
      }
      return ok;
    }
  }

  async function handleMarkPaidSubmit(reference: string) {
    if (!markPaidTarget) return;
    const target = markPaidTarget;
    const ok = await markPaid(target.id, reference);
    if (ok) {
      setMarkPaidTarget(null);
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
            (openRow.status === "pending_review" || openRow.status === "held") &&
            !payoutActionsBlockedByHold(openRow)
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

  const slab = PAYOUT_TAB_SLAB[tab];
  return (
    <>
      <div className="mb-6">
        <label htmlFor="payout-search" className="sr-only">Payout id</label>
        <input
          id="payout-search"
          type="search"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by payout id"
          className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
        />
      </div>
      {list.error ? (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-error">Couldn&apos;t load payouts. Please try again.</p>
          <button type="button" onClick={list.refetch} disabled={list.isFetching} className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}>
            Retry
          </button>
        </div>
      ) : null}
      {/* No count when the counts route is missing or loading: a 0 would read "nothing to do". */}
      <AdminStatusTabs
        label="Payout status"
        tabs={PAYOUT_TABS.map((key) => ({
          key,
          label: PAYOUT_TAB_LABEL[key],
          count: counts && TAB_STATUSES[key].reduce((n, status) => n + (counts[status] ?? 0), 0),
        }))}
        value={tab}
        onChange={setTab}
      />
      {list.isLoading ? (
        <p role="status" className="font-sans text-sm text-slate">Loading payouts…</p>
      ) : !(list.error && effective.length === 0) && (
        <PayoutSlab
          id={tab}
          number={slab.number}
          title={PAYOUT_TAB_LABEL[tab]}
          caption={slab.caption}
          totalCount={list.total}
          rows={effective}
          empty={q ? "No payouts in this tab match this payout id." : slab.empty}
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
            shown: effective.length,
            total: list.total,
            increment: 50,
            isLoading: list.isFetchingNextPage,
            onLoadMore: list.fetchNextPage,
          }}
        />
      )}

      {/* Room under the last row, so the fixed bulk bar never covers its buttons. */}
      {selectionActive && <div aria-hidden="true" className="h-48" />}
      <AdminPayoutBulkBar
        count={selected.size}
        totalAmount={totalSelected}
        approveDisabled={approveDisabled}
        submitting={submitting}
        approveDisabledReason={approveDisabledReason}
        onApprove={() => setBulkApproveOpen(true)}
        onHold={handleBulkHoldRequest}
        onClear={clearSelection}
      />

      <AdminPayoutBulkApproveModal
        open={bulkApproveOpen}
        submitting={submitting}
        onCancel={() => setBulkApproveOpen(false)}
        onConfirm={handleBulkApproveConfirm}
        count={selected.size}
        totalAmount={totalSelected}
      />
      {holdTarget && (
        <AdminPayoutHoldModal
          open={true}
          submitting={submitting}
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
          submitting={submitting}
          onClose={() => setMarkPaidTarget(null)}
          onSubmit={handleMarkPaidSubmit}
          cycle={markPaidTarget}
        />
      )}
      {reasonTarget && (
        <AdminPayoutReasonModal
          open={true}
          submitting={submitting}
          kind={reasonTarget.kind}
          payoutId={reasonTarget.cycle.id}
          amount={reasonTarget.cycle.amount}
          onClose={() => setReasonTarget(null)}
          onSubmit={handleReasonSubmit}
        />
      )}
      {confirmTarget && (
        <AdminConfirmModal
          open={true}
          submitting={submitting}
          pendingLabel={confirmTarget.kind === "approve" ? "Approving…" : "Processing…"}
          onClose={() => setConfirmTarget(null)}
          onConfirm={() => void handleConfirmSubmit()}
          title={confirmTarget.kind === "approve" ? "Approve payout" : "Mark processing"}
          confirmLabel={confirmTarget.kind === "approve" ? "Approve" : "Mark processing"}
        >
          <p className="font-sans text-sm text-ink-soft">
            {confirmTarget.kind === "approve" ? "Approve" : "Mark"} payout{" "}
            <span className="text-ink font-mono">{confirmTarget.cycle.id}</span> for{" "}
            <span className="text-ink tnum">{formatPrice(confirmTarget.cycle.amount)}</span>
            {confirmTarget.kind === "approve" ? "?" : " as processing?"}
          </p>
          <p className="font-sans text-sm text-slate">
            {confirmTarget.kind === "approve"
              ? "Approving a payout does not send funds. Requests that need two admins stay pending until a different admin records the second approval."
              : "Do this once the transfer has started. Then mark it paid with the bank reference, or mark it failed."}
          </p>
        </AdminConfirmModal>
      )}

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => { if (!isPending(actionIds)) setRowId(""); }}
          escDisabled={drawerEscDisabled}
          kicker={`Payout · ${openRow.id}`}
          title={openRow.brandName ?? openRow.photographerName}
          rightHeader={
            <AdminStatusPill
              label={ADMIN_PAYOUT_STATUS_LABEL[openRow.status]}
              tone={payoutStatusTone(openRow.status)}
            />
          }
          subtitle={<PayoutSubtitle cycle={openRow} />}
          actions={
            <PayoutDrawerActions
              cycle={openRow}
              submitting={submitting}
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
          {submitting && <p role="status" aria-busy="true">{pendingAction(actionIds)}</p>}
          <PayoutDetailBody cycle={openRow} />
        </AdminDetailDrawer>
      )}
    </>
  );
}

const PAYOUT_TABS = ["pending", "in-progress", "held", "paid", "closed"] as const;
type PayoutTab = (typeof PAYOUT_TABS)[number];
const PAYOUT_TAB_LABEL: Record<PayoutTab, string> = {
  pending: "Pending review",
  "in-progress": "In progress",
  held: "Held",
  paid: "Paid",
  closed: "Closed",
};
const PAYOUT_TAB_OF: Record<AdminPayoutStatus, PayoutTab> = {
  pending_review: "pending", approved: "in-progress", processing: "in-progress", held: "held",
  paid: "paid", failed: "closed", returned: "closed", rejected: "closed", cancelled: "closed",
};
// The status list each tab sends to the server (AX-10's groups).
const TAB_STATUSES = Object.fromEntries(PAYOUT_TABS.map((key) => [key,
  (Object.keys(PAYOUT_TAB_OF) as AdminPayoutStatus[]).filter((status) => PAYOUT_TAB_OF[status] === key)],
)) as Record<PayoutTab, AdminPayoutStatus[]>;
// approved + processing share a tab: both are "reviewed, payment in motion"; Closed keeps
// failed and returned payouts visible next to rejected and cancelled ones.
const PAYOUT_TAB_SLAB: Record<PayoutTab, { number: string; caption: string; empty: string }> = {
  pending: { number: "01", caption: "Awaiting first decision", empty: "No payouts pending review." },
  "in-progress": { number: "02", caption: "Approved · payment in motion", empty: "No payouts in progress." },
  held: { number: "03", caption: "Paused with reason", empty: "No held payouts." },
  paid: { number: "04", caption: "Settled", empty: "No paid payouts yet." },
  closed: { number: "05", caption: "Rejected, cancelled, failed or returned", empty: "No closed payouts." },
};
// Paid and Closed have no bulk action, so their rows get no checkbox.
const SELECTABLE_TABS: readonly string[] = ["pending", "in-progress", "held"];

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
    isLoading?: boolean;
    onLoadMore: () => void;
  };
}) {
  const { isPending, pendingAction } = usePayoutActions();
  const noun = totalCount === 1 ? "payout" : "payouts";
  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${totalCount} ${noun}`}
      collapsed={totalCount === 0}
    >
      {totalCount === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{empty}</p>
      ) : (
        <>
          {/* @container: the table grid follows this box's width, not the viewport, since the admin rail takes room. */}
          <div className="@container border border-line rounded-2xl bg-surface overflow-hidden">
          <PayoutTableHeader />
          <ul className="divide-y divide-line">
            {rows.map((row) => (
              <li key={row.id}>
                <AdminPayoutRow
                  cycle={row}
                  selectable={SELECTABLE_TABS.includes(id)}
                  selected={selected.has(row.id)}
                  selectionActive={selectionActive}
                  submitting={isPending([row.id])}
                  pendingLabel={pendingAction([row.id])}
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
          </div>
          {loadMore && (
            <LoadMoreButton
              shown={loadMore.shown}
              total={loadMore.total}
              increment={loadMore.increment}
              onLoadMore={loadMore.onLoadMore}
              isLoading={loadMore.isLoading}
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
    <p className="font-sans text-sm text-slate-soft tnum break-words">
      {cycle.handle ? `@${cycle.handle}` : "—"}
      <span className="text-slate-soft"> · </span>
      {sentenceCaseDate(formatLongDate(cycle.submittedAt))}
      <span className="text-slate-soft"> · </span>
      {formatPrice(cycle.amount)}
    </p>
  );
}

// Full per-status action matrix — mirrors AdminPayoutService.ensureFrom so a
// shown button never 409s (Approve, Mark processing and Mark paid also go grey
// while the photographer's relevance hold is on; Hold and Reject stay):
//   pending_review → Hold, Reject, Approve      approved → Hold, Reject, Mark processing, Mark paid
//   processing     → Hold, Mark failed, Mark paid   held → Reject, Approve
//   paid           → Mark returned               failed/returned/rejected/cancelled → none
function PayoutDrawerActions({
  cycle,
  submitting,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
  onReject,
  onMarkFailed,
  onMarkReturned,
}: {
  cycle: AdminPayoutCycle;
  submitting: boolean;
  onApprove: () => void;
  onHold: () => void;
  onMarkPaid: () => void;
  onMarkProcessing: () => void;
  onReject: () => void;
  onMarkFailed: () => void;
  onMarkReturned: () => void;
}) {
  // Same verbs, same look as the row: the main step is primary, a step that stops or undoes money is danger.
  const PRIMARY = cn(BTN_PRIMARY, BTN_SIZE.sm);
  const SECONDARY = cn(BTN_SECONDARY, BTN_SIZE.sm);
  const DANGER = cn(BTN_DANGER, BTN_SIZE.sm);

  const s = cycle.status;
  const blocked = submitting || payoutActionsBlockedByHold(cycle);

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
      <button type="button" disabled={submitting} onClick={onMarkReturned} className={DANGER}>
        Mark returned…
      </button>
    );
  }

  return (
    <>
      {(s === "pending_review" || s === "approved" || s === "processing") && (
        <button type="button" disabled={submitting} onClick={onHold} className={DANGER}>
          Hold…
        </button>
      )}
      {(s === "pending_review" || s === "approved" || s === "held") && (
        <button type="button" disabled={submitting} onClick={onReject} className={DANGER}>
          Reject…
        </button>
      )}
      {s === "approved" && (
        <button type="button" disabled={blocked} onClick={onMarkProcessing} className={SECONDARY}>
          Mark processing
        </button>
      )}
      {s === "processing" && (
        <button type="button" disabled={submitting} onClick={onMarkFailed} className={DANGER}>
          Mark failed…
        </button>
      )}
      {(s === "approved" || s === "processing") && (
        <button type="button" disabled={blocked} onClick={onMarkPaid} className={SECONDARY}>
          Mark paid…
        </button>
      )}
      {(s === "pending_review" || s === "held") && (
        <button type="button" disabled={blocked} onClick={onApprove} className={PRIMARY}>
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
      <PayoutFlagLine cycle={cycle} />
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
            value={sentenceCaseDate(formatLongDate(cycle.submittedAt))}
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
              <p className="font-mono text-[13px] text-slate mt-1 break-all">
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
              <p className="font-sans text-sm text-slate-soft mt-2 tnum">
                Held {sentenceCaseDate(formatLongDate(cycle.reviewedAt))}
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
                value={sentenceCaseDate(formatLongDate(cycle.paidAt))}
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
            Approved {sentenceCaseDate(formatLongDate(cycle.reviewedAt))} — awaiting payment.
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
      <Kicker as="dt" tone="soft">
        {label}
      </Kicker>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}
