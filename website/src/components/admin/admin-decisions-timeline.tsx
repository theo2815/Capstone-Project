"use client";

import { useMemo } from "react";
import {
  ADMIN_USER_SEED,
  type AdminUserRow,
} from "@/lib/admin-user-registry";
import {
  useAdminUserStore,
  type DecisionLogEntry,
  type DecisionType,
} from "@/store/admin-user-store";

interface AdminDecisionsTimelineProps {
  /** Cap on entries rendered. Defaults to 10 (Overview compact view). */
  limit?: number;
  /** Filter to a single user — used by the photographer detail page. */
  userId?: string;
  /** Empty-state copy. */
  emptyCopy?: string;
}

// Timeline of admin decisions read from useAdminUserStore.log. Each row
// shows: action chip · target name · timestamp · optional reason / meta.
// The action chip is the only colored element; everything else stays
// slate/ink. Mirrors the transaction-ledger pattern from /dashboard/billing.
export function AdminDecisionsTimeline({
  limit = 10,
  userId,
  emptyCopy = "No decisions on file yet.",
}: AdminDecisionsTimelineProps) {
  const overrides = useAdminUserStore((s) => s.overrides);
  const log = useAdminUserStore((s) => s.log);

  const byId = useMemo(() => {
    const map = new Map<string, AdminUserRow>();
    for (const row of ADMIN_USER_SEED) {
      const patch = overrides[row.userId];
      map.set(row.userId, patch ? { ...row, ...patch } : row);
    }
    return map;
  }, [overrides]);

  const visible = useMemo(() => {
    const filtered = userId ? log.filter((e) => e.userId === userId) : log;
    return filtered.slice(0, limit);
  }, [log, userId, limit]);

  if (visible.length === 0) {
    return <p className="font-sans text-sm text-slate-soft">{emptyCopy}</p>;
  }

  return (
    <ul className="space-y-3">
      {visible.map((entry, i) => {
        const target = byId.get(entry.userId);
        const targetLabel = target?.brandName ?? target?.name ?? entry.userId;
        return (
          <li
            key={`${entry.userId}-${entry.decidedAt}-${i}`}
            className="flex items-start justify-between gap-4 border-b border-line pb-3"
          >
            <div className="min-w-0 flex-1">
              <p className="font-display text-base text-ink truncate">
                {targetLabel}
              </p>
              <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1 tnum">
                {formatDecidedAt(entry.decidedAt)}
                {renderMetaSuffix(entry)}
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
        );
      })}
    </ul>
  );
}

const DECISION_LABEL: Record<DecisionType, string> = {
  approved: "Approved",
  rejected: "Rejected",
  suspended: "Suspended",
  unsuspended: "Unsuspended",
  "force-edited": "Force-edited",
  reset: "Reset",
};

function decisionTone(decision: DecisionType): string {
  switch (decision) {
    case "approved":
      return "text-fresh";
    case "suspended":
      return "text-ink";
    default:
      return "text-slate";
  }
}

function renderMetaSuffix(entry: DecisionLogEntry): string {
  if (!entry.meta) return "";
  const parts: string[] = [];
  if (entry.meta.field) parts.push(entry.meta.field);
  if (entry.meta.from && entry.meta.to) {
    parts.push(`${entry.meta.from} → ${entry.meta.to}`);
  }
  if (parts.length === 0) return "";
  return ` · ${parts.join(" · ")}`;
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
