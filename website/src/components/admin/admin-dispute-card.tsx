"use client";

import Link from "next/link";
import {
  type Dispute,
  type DisputeStatus,
  DISPUTE_REASON_LABEL,
} from "@/lib/admin-disputes";
import { ROUTES } from "@/lib/constants";
import { formatPrice } from "@/lib/utils";
import { AdminStatusPill, type AdminStatusPillTone } from "./admin-status-pill";

interface AdminDisputeCardProps {
  dispute: Dispute;
  /**
   * Phase 3: when provided, the card behaves as a click-to-open trigger
   * for the drawer host. The default (no callback) preserves the legacy
   * Link wrap to /admin/disputes/[id], so the dedicated detail route
   * still acts as a deep-link backup if you visit it directly.
   */
  onOpen?: () => void;
  /** Phase 4: J/K keyboard focus indicator. Adds an ink ring + offset. */
  focused?: boolean;
}

const STATUS_LABEL: Record<DisputeStatus, string> = {
  open: "Open",
  resolved: "Resolved",
  denied: "Denied",
  escalated: "Escalated",
};

function statusTone(status: DisputeStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "resolved":
      return "fresh";
    case "denied":
      return "ink";
    case "escalated":
      return "slate";
  }
}

// Directory list-row for /admin/disputes. Whole card is a Link to the
// detail page; no nested interactives. Left: identity + reason chip.
// Right: status pill + age + amount.
export function AdminDisputeCard({
  dispute,
  onOpen,
  focused = false,
}: AdminDisputeCardProps) {
  const showAmount =
    dispute.status === "open" || dispute.status === "escalated"
      ? dispute.orderSnapshot.total
      : dispute.refundAmount ?? dispute.orderSnapshot.total;

  const inner = (
    <article
      data-row-id={dispute.id}
      className={`rounded-2xl border border-line bg-bone p-5 md:p-6 transition-all duration-300 hover:border-ink hover:-translate-y-[2px] ${
        focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""
      }`}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-3 flex-wrap">
            <h3 className="font-display text-lg md:text-xl font-medium text-ink truncate">
              @{dispute.runnerHandle}
              <span className="text-slate-soft"> vs </span>
              @{dispute.photographerHandle}
            </h3>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
              {dispute.id}
            </p>
          </div>
          <p className="font-sans text-sm text-slate mt-2 truncate">
            {DISPUTE_REASON_LABEL[dispute.reason]}
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1 tnum">
            Order {dispute.orderId}
            <span className="text-slate-soft"> · </span>
            {formatAge(dispute.reportedAt)}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0 ml-auto md:ml-0">
          <AdminStatusPill
            label={STATUS_LABEL[dispute.status]}
            tone={statusTone(dispute.status)}
          />
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink tnum">
            {formatPrice(showAmount)}
          </p>
        </div>
      </div>
    </article>
  );

  if (onOpen) {
    return (
      <button
        type="button"
        onClick={onOpen}
        aria-label={`Open dispute ${dispute.id}`}
        className="block w-full text-left"
      >
        {inner}
      </button>
    );
  }

  const href = `${ROUTES.ADMIN_DISPUTES}/${dispute.id}`;
  return (
    <Link href={href} className="block">
      {inner}
    </Link>
  );
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
