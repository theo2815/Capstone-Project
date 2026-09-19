"use client";

import Link from "next/link";
import { useAdminUserDetail } from "@/hooks/use-admin-users";
import type { DecisionLogEntry, DecisionType } from "@/lib/admin-user-registry";

interface AdminDecisionsTimelineProps {
  /** Cap on entries rendered. Defaults to 10 (Overview compact view). */
  limit?: number;
  /** Filter to a single user — used by the photographer detail page. */
  userId?: string;
  /** Empty-state copy. */
  emptyCopy?: string;
}

export function AdminDecisionsTimeline({
  limit = 10,
  userId,
  emptyCopy = "No decisions on file yet.",
}: AdminDecisionsTimelineProps) {
  const query = useAdminUserDetail(userId ?? null);
  if (!userId) return <p className="font-sans text-sm text-slate-soft">Decision history is available on each <Link className="underline" href="/admin/photographers">photographer’s profile</Link>.</p>;
  if (query.isPending) return <p role="status">Loading decisions…</p>;
  if (query.isError) return <p role="alert">Couldn&apos;t load decisions. <button onClick={() => void query.refetch()}>Try again</button></p>;
  const visible = (query.data?.decisionLog ?? []).slice(0, limit);

  if (visible.length === 0) {
    return <p className="font-sans text-sm text-slate-soft">{emptyCopy}</p>;
  }

  return (
    <ul className="space-y-3">
      {visible.map((entry, i) => {
        const target = query.data;
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
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1 tnum">
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
              className={`font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] shrink-0 mt-1 ${decisionTone(entry.decision)}`}
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
  messaged: "Messaged",
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
