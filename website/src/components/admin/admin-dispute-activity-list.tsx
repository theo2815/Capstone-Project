"use client";

import { useMemo } from "react";
import {
  useAdminDisputeStore,
  type DisputeDecision,
} from "@/store/admin-dispute-store";
import { DISPUTE_RESOLUTION_LABEL } from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";

interface AdminDisputeActivityListProps {
  disputeId: string;
}

// Per-dispute activity log. Mirrors the photographer-domain timeline but
// reads from useAdminDisputeStore.log and renders dispute-specific meta
// (resolution + refund amount where applicable). Empty state copy is
// generic — admins land here on every detail page even if there's no
// activity yet.
export function AdminDisputeActivityList({
  disputeId,
}: AdminDisputeActivityListProps) {
  const log = useAdminDisputeStore((s) => s.log);

  const visible = useMemo(
    () => log.filter((e) => e.disputeId === disputeId).slice(0, 20),
    [log, disputeId],
  );

  if (visible.length === 0) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        No actions on file yet. Resolve, deny, or escalate to start the
        timeline.
      </p>
    );
  }

  return (
    <ul className="space-y-3">
      {visible.map((entry, i) => (
        <li
          key={`${entry.disputeId}-${entry.decidedAt}-${i}`}
          className="flex items-start justify-between gap-4 border-b border-line pb-3"
        >
          <div className="min-w-0 flex-1">
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
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
            className={`font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] shrink-0 mt-1 ${decisionTone(entry.decision)}`}
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
};

function decisionTone(decision: DisputeDecision): string {
  switch (decision) {
    case "resolved":
      return "text-fresh";
    case "denied":
      return "text-ink";
    case "escalated":
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
