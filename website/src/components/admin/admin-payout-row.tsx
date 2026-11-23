"use client";

import Link from "next/link";
import { useId } from "react";
import {
  type AdminPayoutCycle,
  type AdminPayoutStatus,
  ADMIN_PAYOUT_STATUS_LABEL,
  PAYOUT_HOLD_REASON,
  PAYOUT_UNCONFIRMED_FLAG_NOTE,
  payoutActionsBlockedByHold,
  payoutMethodLabel,
} from "@/lib/admin-payouts";
import { cn, formatPrice } from "@/lib/utils";
import { BTN_DANGER, BTN_SECONDARY, ROW_OPEN_BUTTON } from "@/components/ui/button-styles";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { type AdminStatusPillTone } from "./admin-status-pill";

interface AdminPayoutRowProps {
  cycle: AdminPayoutCycle;
  /** False on tabs with no bulk action (Paid, Closed): no checkbox. */
  selectable?: boolean;
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

/** Status colour for payout rows and the payout drawer pill (one helper for both). */
export function payoutStatusTone(status: AdminPayoutStatus): AdminStatusPillTone {
  switch (status) {
    case "pending_review":
      return "amber";
    // FT-16 (C-25): no status is fresh; green would read as finished on a tab of approved rows.
    case "approved":
    case "processing":
    case "held":
      return "ink";
    case "paid":
      return "slate";
    case "failed":
    case "returned":
      return "error";
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
// The title button covers the row (ROW_OPEN_BUTTON), so a click anywhere
// calls onActivate: with selectionActive=false it opens the focus-mode
// drawer; with selectionActive=true it toggles the checkbox (bulk triage
// takes over). The chevron on the right always opens the drawer regardless
// of selection state. Inner controls sit above the cover (relative z-10).
export function AdminPayoutRow({
  cycle,
  selectable = true,
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
  const hintId = useId();

  return (
    <article
      data-row-id={cycle.id}
      className={cn(
        "group relative cursor-pointer px-4 py-3 transition-colors",
        // One table row: stacked while the table is narrow, the shared column grid once it is wide (@5xl, a container query).
        "flex flex-wrap items-start gap-x-4 gap-y-2",
        "@5xl:grid @5xl:items-center @5xl:gap-x-4",
        PAYOUT_TABLE_COLUMNS,
        selected ? "bg-fresh-tint/50" : "hover:bg-bone-deep/50",
        focused && "ring-2 ring-inset ring-ink",
      )}
    >
      {selectable ? (
        <input
          type="checkbox"
          aria-label={`Select payout ${cycle.id}`}
          checked={selected}
          disabled={submitting}
          onChange={onToggleSelect}
          className="relative z-10 mt-1 @5xl:mt-0 accent-ink shrink-0 size-4 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        />
      ) : (
        <span aria-hidden="true" className="hidden @5xl:block" />
      )}
      <div className="min-w-0 flex-1 basis-[calc(100%-2rem)] @5xl:basis-auto">
        <h3 className="font-display text-base font-semibold text-ink truncate">
          <button type="button" onClick={onActivate} aria-describedby={hintId} className={cn(ROW_OPEN_BUTTON, "block w-full truncate")}>
            {subjectLabel}
          </button>
        </h3>
        <span id={hintId} className="sr-only">{selectionActive ? "Toggles selection" : "Opens the payout"}</span>
        <p className="font-sans text-sm text-slate truncate">
          {cycle.handle && <span className="font-mono text-[13px]">@{cycle.handle}</span>}
          {cycle.handle && " · "}
          <span className="font-mono text-[13px]">{cycle.id}</span>
        </p>
        <PayoutFlagLine cycle={cycle} />
        {cycle.status === "held" && cycle.holdReason && (
          <p className="font-sans text-sm text-slate mt-1">Hold: {cycle.holdReason}</p>
        )}
        {cycle.status === "rejected" && cycle.rejectedReason && (
          <p className="font-sans text-sm text-slate mt-1">Rejected: {cycle.rejectedReason}</p>
        )}
        {cycle.status === "failed" && cycle.failedReason && (
          <p className="font-sans text-sm text-slate mt-1">Failed: {cycle.failedReason}</p>
        )}
        {cycle.status === "paid" && cycle.paymentReference && (
          <p className="font-sans text-sm text-slate mt-1">
            Ref <span className="font-mono text-[13px] text-ink">{cycle.paymentReference}</span>
            {cycle.paidAt && ` · paid ${sentenceCaseDate(formatLongDate(cycle.paidAt))}`}
          </p>
        )}
      </div>
      {/* Request time, matching the photographer's own view: see the note on
          PayoutSubtitle in payouts-queue.tsx. */}
      <Cell label="Requested" className="text-ink-soft whitespace-nowrap">{sentenceCaseDate(formatLongDate(cycle.submittedAt))}</Cell>
      <Cell label="Sales" className="@5xl:text-right font-mono tnum text-[13px] text-ink-soft">{cycle.itemCount}</Cell>
      <Cell label="Amount" className="@5xl:text-right font-mono tnum text-[14px] font-medium text-ink">{formatPrice(cycle.amount)}</Cell>
      <Cell label="Method" className="text-ink-soft">{payoutMethodLabel(cycle.method)}</Cell>
      <Cell label="Status">
        <StatusDot status={cycle.status} />
      </Cell>
      <div className="flex items-center justify-end gap-1 w-full min-w-0 @5xl:w-auto">
        {submitting && <p role="status" aria-busy="true" className="font-sans text-sm text-slate mr-2">{pendingLabel}</p>}
        <RowActions
          submitting={submitting}
          status={cycle.status}
          provider={cycle.provider ?? "manual"}
          blocked={payoutActionsBlockedByHold(cycle)}
          onApprove={onApprove}
          onHold={onHold}
          onMarkPaid={onMarkPaid}
          onMarkProcessing={onMarkProcessing}
        />
        <button
          type="button"
          onClick={onOpenDrawer}
          aria-label={`Open payout ${cycle.id}`}
          className="relative z-10 shrink-0 size-11 rounded-full text-slate hover:text-ink hover:bg-bone-deep transition-colors flex items-center justify-center"
        >
          <span aria-hidden="true" className="text-lg leading-none">
            ›
          </span>
        </button>
      </div>
    </article>
  );
}

/** Column template shared by the payout rows and their header, once the table
 *  is at least 64rem wide (@5xl container query: the fixed tracks alone take
 *  about 45rem, so below that the photographer column collapsed to 0, and
 *  at 56rem it still spilled; measured in AX-29).
 *  Fixed status and actions tracks: an `auto` track sized differently in the
 *  header and in each row, so the headings drifted off their columns. */
export const PAYOUT_TABLE_COLUMNS =
  "@5xl:grid-cols-[1rem_minmax(0,2fr)_minmax(0,1fr)_3.5rem_7.5rem_minmax(0,0.9fr)_8.5rem_16rem]";

/** Column headings for the payout table; each row cell also names itself for screen readers. */
export function PayoutTableHeader() {
  const H = "font-mono uppercase tracking-[0.12em] text-[12px] text-slate";
  return (
    <div aria-hidden="true" className={cn("hidden @5xl:grid @5xl:items-center @5xl:gap-x-4 px-4 py-3 border-b border-line", PAYOUT_TABLE_COLUMNS)}>
      <span />
      <span className={H}>Photographer</span>
      <span className={H}>Requested</span>
      <span className={cn(H, "text-right")}>Sales</span>
      <span className={cn(H, "text-right")}>Amount</span>
      <span className={H}>Method</span>
      <span className={H}>Status</span>
      <span className={cn(H, "text-right pr-12")}>Actions</span>
    </div>
  );
}

function Cell({ label, className, children }: { label: string; className?: string; children: React.ReactNode }) {
  return (
    <div className={cn("font-sans text-sm basis-[45%] @5xl:basis-auto min-w-0", className)}>
      {/* Visible while stacked; in the wide table only screen readers hear it, since the header row names the column. */}
      <span className="@5xl:sr-only font-mono uppercase tracking-[0.12em] text-[12px] text-slate-soft block">{label}</span>
      {children}
    </div>
  );
}

const STATUS_TEXT: Record<AdminStatusPillTone, string> = {
  amber: "text-warning-text",
  fresh: "text-fresh",
  ink: "text-ink",
  slate: "text-slate",
  muted: "text-slate-soft",
  error: "text-error",
};

// A dot plus the word: colour is never the only signal, and it reads quieter
// than an outlined pill inside a table. The dot takes the text colour.
function StatusDot({ status }: { status: AdminPayoutStatus }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-sm whitespace-nowrap", STATUS_TEXT[payoutStatusTone(status)])}>
      <span aria-hidden="true" className="size-[7px] rounded-full bg-current" />
      {ADMIN_PAYOUT_STATUS_LABEL[status]}
    </span>
  );
}

/** The flag note under a payout's id: a hold, or a warning while the flag is unconfirmed. */
export function PayoutFlagLine({ cycle }: { cycle: AdminPayoutCycle }) {
  if (!cycle.relevanceFlagId) return null;
  return (
    <p className="font-sans text-sm text-warning-text mt-2">
      <Link href={`/admin/flags?row=${cycle.relevanceFlagId}`} className="relative z-10 underline underline-offset-2 hover:text-ink">
        {cycle.relevanceHold ? PAYOUT_HOLD_REASON : PAYOUT_UNCONFIRMED_FLAG_NOTE}
      </Link>
    </p>
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
  blocked,
  onApprove,
  onHold,
  onMarkPaid,
  onMarkProcessing,
}: {
  blocked: boolean;
  status: AdminPayoutStatus;
  submitting: boolean;
  provider: string;
  onApprove: () => void;
  onHold: () => void;
  onMarkPaid: () => void;
  onMarkProcessing: () => void;
}) {
  if (provider !== "manual") return null;
  // House buttons in sentence case (proposal 11: mono uppercase is for labels only).
  const PILL = cn(BTN_SECONDARY, "px-4 py-2 text-sm min-h-[44px] whitespace-nowrap");
  const GHOST = cn(BTN_DANGER, "px-4 py-2 text-sm min-h-[44px] whitespace-nowrap");

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
    <div className="relative z-10 min-w-0 flex items-center gap-2 flex-wrap justify-end">
      {canApprove && (
        <button type="button" disabled={submitting || blocked} onClick={onApprove} className={PILL}>
          Approve
        </button>
      )}
      {canMarkProcessing && (
        <button type="button" disabled={submitting || blocked} onClick={onMarkProcessing} className={PILL}>
          Mark processing
        </button>
      )}
      {canMarkPaid && (
        <button type="button" disabled={submitting || blocked} onClick={onMarkPaid} className={PILL}>
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
