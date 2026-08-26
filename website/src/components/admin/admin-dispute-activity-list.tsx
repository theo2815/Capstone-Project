"use client";

import { useMemo } from "react";
import {
  useAdminDisputeStore,
  type DisputeDecision,
} from "@/store/admin-dispute-store";
import {
  DISPUTE_RESOLUTION_LABEL,
  type DisputeActivityEntry,
  type DisputeResolution,
} from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";

interface AdminDisputeActivityListProps {
  disputeId: string;
  // Server-side audit trail from admin_decision_log. When provided, this is
  // the source of truth. Local-store entries layer on top to surface
  // optimistic actions that haven't roundtripped yet.
  activity?: DisputeActivityEntry[];
}

interface ActivityRow {
  key: string;
  decidedAt: string;
  decision: DisputeDecision;
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  reason: string | null;
}

// Per-dispute activity log. Prefers the BE-persisted activity (survives
// page reloads + session changes); merges in the local-store optimistic
// log so admin actions show instantly before the query refetches.
export function AdminDisputeActivityList({
  disputeId,
  activity,
}: AdminDisputeActivityListProps) {
  const localLog = useAdminDisputeStore((s) => s.log);

  const rows = useMemo<ActivityRow[]>(() => {
    const serverRows: ActivityRow[] = (activity ?? []).map((e) => ({
      key: `srv-${e.id}`,
      decidedAt: e.decidedAt,
      decision: e.decision,
      resolution: e.resolution,
      refundAmount: e.refundAmount,
      reason: e.reason,
    }));

    // Layer local entries on top — keyed by decidedAt to avoid double-counting
    // once the BE round-trips and the same action lands in `activity`. We
    // consider a local entry already represented if any server row's
    // decidedAt is within 60s of it AND the decisions match.
    const matched = (local: (typeof localLog)[number]): boolean => {
      const lt = Date.parse(local.decidedAt);
      if (!Number.isFinite(lt)) return false;
      return serverRows.some((s) => {
        const st = Date.parse(s.decidedAt);
        return (
          s.decision === local.decision &&
          Number.isFinite(st) &&
          Math.abs(st - lt) < 60_000
        );
      });
    };

    const localRows: ActivityRow[] = localLog
      .filter((e) => e.disputeId === disputeId && !matched(e))
      .map((e, i) => ({
        key: `loc-${e.disputeId}-${e.decidedAt}-${i}`,
        decidedAt: e.decidedAt,
        decision: e.decision,
        resolution: e.resolution,
        refundAmount: e.refundAmount,
        reason: e.reason,
      }));

    // Newest first — matches the BE ORDER BY decidedAt DESC.
    return [...serverRows, ...localRows]
      .sort((a, b) => b.decidedAt.localeCompare(a.decidedAt))
      .slice(0, 20);
  }, [activity, localLog, disputeId]);

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
          key={entry.key}
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
