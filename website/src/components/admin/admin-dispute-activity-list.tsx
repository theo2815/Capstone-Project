"use client";

import { useMemo } from "react";
import {
  DISPUTE_RESOLUTION_LABEL,
  type DisputeActivityEntry,
} from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";

type DisputeDecision = DisputeActivityEntry["decision"];

interface AdminDisputeActivityListProps {
  activity?: DisputeActivityEntry[];
}

// Render the backend audit trail directly, without session logs or timestamp deduplication.
export function AdminDisputeActivityList({ activity }: AdminDisputeActivityListProps) {
  const rows = useMemo(
    () => [...(activity ?? [])].sort((a, b) => b.decidedAt.localeCompare(a.decidedAt)).slice(0, 20),
    [activity],
  );

  if (rows.length === 0) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        No actions on file yet. Resolve, deny, or escalate to start the
        timeline.
      </p>
    );
  }

  return (
    <ul className="space-y-3">
      {rows.map((entry) => (
        <li
          key={entry.id}
          className="flex items-start justify-between gap-4 border-b border-line pb-3"
        >
          <div className="min-w-0 flex-1">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
              {formatDecidedAt(entry.decidedAt)}
              {entry.resolution &&
                ` · ${DISPUTE_RESOLUTION_LABEL[entry.resolution]}`}
              {entry.refundAmount !== null &&
                ` · ${formatPrice(entry.refundAmount)}`}
            </p>
            {entry.reason && (
              <p className="font-sans text-sm text-slate mt-1">
                {entry.reason}
              </p>
            )}
          </div>
          <span
            className={`font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] shrink-0 mt-1 ${decisionTone(entry.decision)}`}
          >
            {DECISION_LABEL[entry.decision]}
          </span>
        </li>
      ))}
    </ul>
  );
}

const DECISION_LABEL: Record<DisputeDecision, string> = {
  resolved: "Resolved",
  denied: "Denied",
  escalated: "Escalated",
  refund_sent_to_manual_review: "Refund approved, waiting for PayMongo",
  refund_link_sent: "Refund link sent",
  refund_cancelled: "Refund cancelled: window closed",
};

function decisionTone(decision: DisputeDecision): string {
  switch (decision) {
    case "resolved":
      return "text-fresh";
    case "denied":
    case "refund_cancelled":
      return "text-ink";
    case "escalated":
    case "refund_sent_to_manual_review":
    case "refund_link_sent":
      return "text-slate";
  }
}

function formatDecidedAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const time = d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
  return `${month} ${day} · ${time}`;
}
