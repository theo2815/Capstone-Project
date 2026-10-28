"use client";

import { useAdminUserDetail } from "@/hooks/use-admin-users";
import Link from "next/link";
import { useAdminDecisions } from "@/hooks/use-admin-decisions";
import type { RecentAdminDecision } from "@/lib/api-admin";
import type { DecisionLogEntry } from "@/lib/admin-user-registry";

interface AdminDecisionsTimelineProps {
  /** Cap on entries rendered. Defaults to 10 (Overview compact view). */
  limit?: number;
  /** Filter to a single user — used by the photographer detail page. */
  userId?: string;
  /** Overview only: with no user id, list recent decisions of every kind instead of the placeholder. */
  recent?: boolean;
  /** Empty-state copy. */
  emptyCopy?: string;
}

export function AdminDecisionsTimeline({
  limit = 10,
  userId,
  recent = false,
  emptyCopy = "No decisions on file yet.",
}: AdminDecisionsTimelineProps) {
  const userQuery = useAdminUserDetail(userId ?? null);
  const recentQuery = useAdminDecisions(limit, recent && !userId);
  if (!userId && !recent) return <p className="font-sans text-sm text-slate-soft">Decision history is available on each <Link className="underline" href="/admin/photographers">photographer’s profile</Link>.</p>;
  const query = userId ? userQuery : recentQuery;
  if (query.isPending) return <p role="status">Loading decisions…</p>;
  if (query.isError) return <p role="alert">Couldn&apos;t load decisions. <button onClick={() => void query.refetch()}>Try again</button></p>;

  const rows: TimelineRow[] = userId
    ? (userQuery.data?.decisionLog ?? []).slice(0, limit).map((entry, i) => ({
        key: `${entry.userId}-${entry.decidedAt}-${i}`,
        title: userQuery.data?.brandName ?? userQuery.data?.name ?? entry.userId,
        decision: entry.decision,
        when: `${formatDecidedAt(entry.decidedAt)}${renderMetaSuffix(entry)}`,
        adminName: entry.adminName ?? null,
        reason: entry.reason,
      }))
    : (recentQuery.data ?? []).slice(0, limit).map((d) => ({
        key: d.id,
        title: targetTitle(d),
        decision: d.decision,
        when: formatDecidedAt(d.decidedAt),
        adminName: d.adminName,
        reason: d.reason,
      }));

  if (rows.length === 0) {
    return <p className="font-sans text-sm text-slate-soft">{emptyCopy}</p>;
  }

  return (
    <ul className="space-y-3">
      {rows.map((row) => (
        <li
          key={row.key}
          className="flex items-start justify-between gap-4 border-b border-line pb-3"
        >
          <div className="min-w-0 flex-1">
            <p className="font-display text-base text-ink truncate">
              {row.title}
            </p>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1 tnum">
              {row.when} · by {row.adminName ?? "unknown admin"}
            </p>
            {row.reason && (
              <p className="font-sans text-sm text-slate mt-1">
                {row.reason}
              </p>
            )}
          </div>
          <span
            className={`font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] shrink-0 mt-1 ${decisionTone(row.decision)}`}
          >
            {decisionLabel(row.decision)}
          </span>
        </li>
      ))}
    </ul>
  );
}

interface TimelineRow {
  key: string;
  title: string;
  decision: string;
  when: string;
  adminName: string | null;
  reason: string | null;
}

const DECISION_LABEL: Record<string, string> = {
  approved: "Approved",
  rejected: "Rejected",
  suspended: "Suspended",
  unsuspended: "Unsuspended",
  "force-edited": "Force-edited",
  reset: "Reset",
  messaged: "Messaged",
};

// Known user decisions keep their label; other codes (PAYOUT_MARKED_PAID) read as a sentence.
function decisionLabel(code: string): string {
  const known = DECISION_LABEL[code];
  if (known) return known;
  const words = code.toLowerCase().replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const TARGET_NAME = { user: "User", payout: "Payout", dispute: "Dispute", event: "Event", removal: "Photo removal" } as const;

function targetTitle(d: RecentAdminDecision): string {
  if (d.targetName) return d.targetName;
  if (!d.targetType) return "Admin action";
  // UUIDs are long; their first 8 characters are enough to tell rows apart.
  const id = d.targetId && d.targetId.length > 20 ? d.targetId.slice(0, 8) : (d.targetId ?? "");
  return `${TARGET_NAME[d.targetType]} ${id}`.trim();
}

function decisionTone(decision: string): string {
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
