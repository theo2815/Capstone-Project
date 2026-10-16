"use client";

import {
  type AdminPayoutCycle,
  type AdminPayoutStatus,
  ADMIN_PAYOUT_STATUS_LABEL,
  payoutMethodLabel,
} from "@/lib/admin-payouts";
import { formatPrice } from "@/lib/utils";
import { formatLongDate } from "@/lib/format";
import { AdminStatusPill, type AdminStatusPillTone } from "./admin-status-pill";

interface AdminPayoutRowProps {
  cycle: AdminPayoutCycle;
  submitting: boolean;
  pendingLabel?: string | null;
  selected: boolean;
  selectionActive: boolean;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused: boolean;
  onToggleSelect: () => void;
  onActivate: () => void;
  onOpenDrawer: () => void;
  onApprove: () => void;
  onHold: () => void;
  onMarkPaid: () => void;
  onMarkProcessing: () => void;
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

// Single payout-cycle row for /admin/payouts. Carries a checkbox + the
// per-row action buttons that depend on current status:
//   pending_review → [Approve] [Hold]
//   approved       → [Mark paid] [Hold]
//   held           → [Approve] [Mark paid]
//   paid           → no actions, payment reference visible
//
// Phase 3 makes the row body itself clickable. With selectionActive=false
// the click opens the focus-mode drawer; with selectionActive=true it
// toggles the checkbox (bulk triage takes over). The chevron on the
// right always opens the drawer regardless of selection state. Inline
// action buttons stopPropagation so they don't trigger the row click.
export function AdminPayoutRow({
  cycle,
  submitting,
  pendingLabel,
  selected,
  selectionActive,
  focused,
  onToggleSelect,
  onActivate,
  onOpenDrawer,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
}: AdminPayoutRowProps) {
  const subjectLabel = cycle.brandName ?? cycle.photographerName;

  function handleArticleKeyDown(e: React.KeyboardEvent<HTMLElement>) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onActivate();
    }
  }

  return (
    <article
      role="button"
      tabIndex={0}
      data-row-id={cycle.id}
      onClick={onActivate}
      onKeyDown={handleArticleKeyDown}
      aria-label={`${subjectLabel}${
        selectionActive ? " — toggle selection" : " — open payout"
      }`}
      className={`group rounded-2xl border p-5 md:p-6 transition-colors cursor-pointer ${
        selected
          ? "border-ink bg-fresh-tint/30"
          : "border-line bg-bone hover:border-ink"
      } ${focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""}`}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        <input
          type="checkbox"
          aria-label={`Select payout ${cycle.id}`}
          checked={selected}
          disabled={submitting}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
          className="mt-1 accent-ink shrink-0 size-4 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h3 className="font-display text-lg md:text-xl font-medium text-ink truncate">
              {subjectLabel}
            </h3>
            {cycle.handle && (
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
                @{cycle.handle}
              </p>
            )}
          </div>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-2 tnum">
            {cycle.id}
            <span className="text-slate-soft"> · </span>
            {/* Request time, matching the photographer's own view — see the
                note on PayoutSubtitle in payouts-queue.tsx. */}
            {formatLongDate(cycle.submittedAt)}
            <span className="text-slate-soft"> · </span>
            {cycle.itemCount} {cycle.itemCount === 1 ? "sale" : "sales"}
          </p>
          {cycle.status === "held" && cycle.holdReason && (
            <p className="font-sans text-sm text-slate mt-2">
              <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft mr-2">
                Hold:
              </span>
              {cycle.holdReason}
            </p>
          )}
          {cycle.status === "rejected" && cycle.rejectedReason && (
            <p className="font-sans text-sm text-slate mt-2">
              <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft mr-2">
                Rejected:
              </span>
              {cycle.rejectedReason}
            </p>
          )}
          {cycle.status === "failed" && cycle.failedReason && (
            <p className="font-sans text-sm text-slate mt-2">
              <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft mr-2">
                Failed:
              </span>
              {cycle.failedReason}
            </p>
          )}
          {cycle.status === "paid" && cycle.paymentReference && (
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mt-2 tnum">
              Ref: {cycle.paymentReference}
              {cycle.paidAt && (
                <>
                  <span className="text-slate-soft"> · </span>
                  {formatLongDate(cycle.paidAt)}
                </>
              )}
            </p>
          )}
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0 ml-auto md:ml-0">
          <AdminStatusPill
            label={ADMIN_PAYOUT_STATUS_LABEL[cycle.status]}
            tone={statusTone(cycle.status)}
          />
          <p className="font-display text-lg text-ink tnum">
            {formatPrice(cycle.amount)}
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft">
            {payoutMethodLabel(cycle.method)}
          </p>
        </div>
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onOpenDrawer();
          }}
          aria-label={`Open payout ${cycle.id}`}
          className="shrink-0 size-8 rounded-full text-slate hover:text-ink hover:bg-bone-deep transition-colors flex items-center justify-center"
        >
          <span aria-hidden="true" className="text-lg leading-none">
            ›
          </span>
        </button>
      </div>

      {submitting && <p role="status" aria-busy="true">{pendingLabel}</p>}
      <RowActions
        submitting={submitting}
        status={cycle.status}
        provider={cycle.provider ?? "manual"}
        onApprove={(e) => {
          e.stopPropagation();
          onApprove();
        }}
        onHold={(e) => {
          e.stopPropagation();
          onHold();
        }}
        onMarkPaid={(e) => {
          e.stopPropagation();
          onMarkPaid();
        }}
        onMarkProcessing={(e) => {
          e.stopPropagation();
          onMarkProcessing();
        }}
      />
    </article>
  );
}

// Inline row actions are the PRIMARY next step per status; the full matrix
// (reject / mark-failed / mark-returned) lives in the drawer. Buttons match the
// backend's ensureFrom guards exactly so a shown button never 409s:
//   pending_review → Approve, Hold          approved → Mark processing, Mark paid, Hold
//   processing     → Mark paid, Hold        held     → Approve
//   terminal (paid/failed/returned/rejected/cancelled) → none
function RowActions({
  submitting,
  status,
  provider,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
}: {
  status: AdminPayoutStatus;
  submitting: boolean;
  provider: string;
  onApprove: (e: React.MouseEvent) => void;
  onHold: (e: React.MouseEvent) => void;
  onMarkPaid: (e: React.MouseEvent) => void;
  onMarkProcessing: (e: React.MouseEvent) => void;
}) {
  if (provider !== "manual") return null;
  const PILL =
    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink border border-line hover:bg-ink hover:text-surface hover:border-ink transition-colors rounded-full px-4 py-1.5 disabled:opacity-50 disabled:cursor-not-allowed";
  const GHOST =
    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-3 py-1.5 disabled:opacity-50 disabled:cursor-not-allowed";

  const canApprove = status === "pending_review" || status === "held";
  const canMarkProcessing = status === "approved";
  const canMarkPaid = status === "approved" || status === "processing";
  const canHold =
    status === "pending_review" ||
    status === "approved" ||
    status === "processing";

  if (!canApprove && !canMarkProcessing && !canMarkPaid && !canHold) {
    return null;
  }
  return (
    <div className="mt-4 flex items-center gap-2 flex-wrap">
      {canApprove && (
        <button type="button" disabled={submitting} onClick={onApprove} className={PILL}>
          Approve
        </button>
      )}
      {canMarkProcessing && (
        <button type="button" disabled={submitting} onClick={onMarkProcessing} className={PILL}>
          Mark processing
        </button>
      )}
      {canMarkPaid && (
        <button type="button" disabled={submitting} onClick={onMarkPaid} className={PILL}>
          Mark paid…
        </button>
      )}
      {canHold && (
        <button type="button" disabled={submitting} onClick={onHold} className={GHOST}>
          Hold…
        </button>
      )}
    </div>
  );
}
