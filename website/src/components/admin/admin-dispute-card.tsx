"use client";

import { useId } from "react";
import { Kicker } from "@/components/ui/kicker";
import {
  type Dispute,
  DISPUTE_ATTENTION_LABEL,
  DISPUTE_PAYMENT_LABEL,
  DISPUTE_REASON_LABEL,
  DISPUTE_STATUS_LABEL,
  disputeStatusTone,
} from "@/lib/admin-disputes";
import { formatSalesDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";
import { ROW_OPEN_BUTTON } from "@/components/ui/button-styles";
import { AdminStatusPill } from "./admin-status-pill";

interface AdminDisputeCardProps {
  dispute: Dispute;
  /** Opens the drawer host for this dispute. */
  onOpen: () => void;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

// Directory list-row for /admin/disputes. The title button covers the card
// (ROW_OPEN_BUTTON) and opens the drawer. Left: identity + reason chip.
// Right: status pill + age + amount.
export function AdminDisputeCard({
  dispute,
  onOpen,
  focused = false,
}: AdminDisputeCardProps) {
  // Undecided: the photo's own charge. Resolved: what was refunded, so a partial refund shows its amount.
  // Denied or withdrawn: no money moved. A missing amount falls back to the order total, labelled as such.
  const noRefund = dispute.status === "denied" || dispute.status === "withdrawn";
  const amount = dispute.status === "open" || dispute.status === "escalated" ? dispute.itemChargeAmount : dispute.refundAmount;
  const amountText = noRefund
    ? "No refund"
    : amount != null ? formatPrice(amount) : `Order total ${formatPrice(dispute.orderSnapshot.total)}`;
  const paymentLabel =
    DISPUTE_PAYMENT_LABEL[dispute.orderSnapshot.paymentMethod] ??
    dispute.orderSnapshot.paymentMethod;
  const attentionLabel = dispute.attention ? DISPUTE_ATTENTION_LABEL[dispute.attention] : null;

  const idLineId = useId();

  return (
    <article
      data-row-id={dispute.id}
      className={`relative rounded-2xl border border-line bg-bone p-5 md:p-6 transition-all duration-300 hover:border-ink hover:-translate-y-[2px] ${
        focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""
      }`}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h3 className="font-display text-lg md:text-xl font-medium text-ink truncate">
              <button type="button" onClick={onOpen} aria-describedby={idLineId} className={cn(ROW_OPEN_BUTTON, "block w-full truncate after:rounded-2xl")}>
                @{dispute.runnerHandle}
                <span className="text-slate-soft"> vs </span>
                @{dispute.photographerHandle}
              </button>
            </h3>
            <Kicker as="p" id={idLineId} className="tnum">
              {dispute.id}
            </Kicker>
          </div>
          <p className="font-sans text-sm text-slate mt-2 truncate">
            {DISPUTE_REASON_LABEL[dispute.reason]}
          </p>
          <Kicker as="p" tone="soft" className="mt-1 tnum">
            Order {dispute.orderId}
            <span className="text-slate-soft"> · </span>
            {formatAge(dispute.reportedAt)}
          </Kicker>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0 ml-auto md:ml-0">
          <AdminStatusPill
            label={DISPUTE_STATUS_LABEL[dispute.status]}
            tone={disputeStatusTone(dispute.status)}
          />
          <Kicker as="p" className="text-ink tnum">
            {amountText}
            <span className="text-slate-soft"> · {paymentLabel}</span>
          </Kicker>
        </div>
      </div>
      {(attentionLabel || dispute.dueAt) && (
        // Full card width, so the chips fit at 375 px.
        <div className="flex flex-wrap gap-2 mt-4">
          {attentionLabel && <AdminStatusPill label={attentionLabel} tone="ink" className={CHIP} />}
          {dispute.dueAt && (dispute.overdue
            ? <AdminStatusPill label={`Overdue · ${formatSpan(Date.now() - new Date(dispute.dueAt).getTime())}`} tone="error" className={CHIP} />
            : <AdminStatusPill label={`Due ${formatSalesDate(dispute.dueAt)}`} tone="muted" className={CHIP} />)}
          {dispute.linkExpiringSoon && dispute.refundLinkExpiresAt && (
            <AdminStatusPill label={`Link expires ${formatSalesDate(dispute.refundLinkExpiresAt)}`} tone="amber" className={CHIP} />
          )}
        </div>
      )}
    </article>
  );
}

// Mono chips must not wrap mid-word at 375 px (ui-pitfalls 2026-05-06): one line, cut with an ellipsis.
const CHIP = "block max-w-full truncate tnum";

function formatSpan(ms: number): string {
  const minutes = Math.max(Math.floor(ms / 60000), 1);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`;
}

function formatAge(iso: string): string {
  const then = new Date(iso).getTime();
  if (!Number.isFinite(then)) return "—";
  const diff = Date.now() - then;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}
