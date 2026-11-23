"use client";

import { useMemo, useState } from "react";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import {
  DISPUTE_RESOLUTION_LABEL,
  type DisputeActivityEntry,
} from "@/lib/admin-disputes";
import { formatSalesDate, sentenceCaseDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";

type DisputeDecision = DisputeActivityEntry["decision"];

interface AdminDisputeActivityListProps {
  activity?: DisputeActivityEntry[];
}

const FIRST_ROWS = 20;

// Render the backend audit trail directly, without session logs or timestamp deduplication.
// The newest 20 show first; "Show all" reveals the rest, so no entry is hidden silently.
export function AdminDisputeActivityList({ activity }: AdminDisputeActivityListProps) {
  const [showAll, setShowAll] = useState(false);
  const rows = useMemo(
    () => [...(activity ?? [])].sort((a, b) => b.decidedAt.localeCompare(a.decidedAt)),
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
    <>
    <ul className="space-y-3">
      {(showAll ? rows : rows.slice(0, FIRST_ROWS)).map((entry) => (
        <li key={entry.id} className="border-b border-line pb-3">
          <div className="min-w-0">
            <p className={`font-sans text-sm ${decisionTone(entry.decision)}`}>
              {DECISION_LABEL[entry.decision]}
            </p>
            <p className="font-sans text-sm text-slate-soft tnum break-words mt-1">
              {sentenceCaseDate(formatSalesDate(entry.decidedAt))}
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
        </li>
      ))}
    </ul>
    {!showAll && rows.length > FIRST_ROWS && (
      <button type="button" onClick={() => setShowAll(true)} className={`${BTN_SECONDARY} ${BTN_SIZE.sm} mt-4`}>
        Show all <span className="tnum">{rows.length}</span>
      </button>
    )}
    </>
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
