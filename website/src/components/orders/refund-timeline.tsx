"use client";

import { Kicker } from "@/components/ui/kicker";
import {
  DISPUTE_REASON_LABEL,
  type DisputeReason,
} from "@/lib/admin-disputes";
import type { RunnerDispute, RunnerDisputeStatus } from "@/lib/api-orders";
import { cn, formatPrice } from "@/lib/utils";

interface RefundTimelineProps {
  disputes: ReadonlyArray<RunnerDispute>;
  className?: string;
}

// Per-dispute lifecycle log — one row per Dispute on the order. Editorial
// not chronological: timestamps are mono+tnum, status carries a coloured
// dot, the admin's resolution note (if any) closes the row in italics.
// Withdrawn disputes show — the runner asked to see what they've done.
export function RefundTimeline({ disputes, className }: RefundTimelineProps) {
  if (disputes.length === 0) return null;

  const ordered = [...disputes].sort((a, b) =>
    b.openedAt.localeCompare(a.openedAt),
  );

  return (
    <section
      aria-label="Refund history"
      className={cn(
        "border border-line rounded-xl bg-bone-deep/30",
        className,
      )}
    >
      <header className="px-4 md:px-5 py-3 border-b border-line/60">
        <Kicker as="p">Refund history</Kicker>
      </header>
      <ul className="divide-y divide-line/60">
        {ordered.map((d) => (
          <li key={d.id} className="px-4 md:px-5 py-4">
            <RefundTimelineRow dispute={d} />
          </li>
        ))}
      </ul>
    </section>
  );
}

function RefundTimelineRow({ dispute }: { dispute: RunnerDispute }) {
  const statusBadge = STATUS_BADGE[dispute.status];

  return (
    <div className="space-y-2.5">
      <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <div className="flex items-center gap-2 min-w-0">
          <span
            aria-hidden="true"
            className={cn("size-1.5 rounded-full shrink-0", statusBadge.dot)}
          />
          <span
            className={cn(
              "font-mono uppercase tracking-[0.2em] text-[12px] tnum truncate",
              statusBadge.text,
            )}
          >
            {statusBadge.label}
          </span>
          {dispute.status === "resolved" && dispute.refundAmount !== null && (
            <span className="font-mono tnum text-[12px] text-ink whitespace-nowrap">
              · {formatPrice(dispute.refundAmount)} refunded
            </span>
          )}
        </div>
        <Kicker as="span" tnum>
          Photo {dispute.photoId.replace(/^mock-/, "").slice(0, 8).toUpperCase()}
        </Kicker>
      </header>

      <ol className="font-sans text-sm text-slate space-y-1 pl-3.5">
        <TimelineLine label="Filed" timestamp={dispute.openedAt} />
        {dispute.status === "escalated" && (
          <TimelineLine label="Escalated for review" timestamp={null} muted />
        )}
        {dispute.resolvedAt &&
          (dispute.status === "resolved" || dispute.status === "denied") && (
            <TimelineLine
              label={dispute.status === "resolved" ? "Resolved" : "Denied"}
              timestamp={dispute.resolvedAt}
            />
          )}
        {dispute.withdrawnAt && dispute.status === "withdrawn" && (
          <TimelineLine
            label="You cancelled this request"
            timestamp={dispute.withdrawnAt}
            muted
          />
        )}
      </ol>

      <div className="space-y-1.5 pl-3.5">
        <p className="font-sans text-sm text-ink-soft">
          <span className="text-slate">Reason · </span>
          {DISPUTE_REASON_LABEL[dispute.reason as DisputeReason] ?? dispute.reason}
        </p>
        {dispute.note && (
          <p className="font-sans text-sm text-ink-soft italic">
            &ldquo;{dispute.note}&rdquo;
          </p>
        )}
        {dispute.resolutionNote &&
          (dispute.status === "resolved" || dispute.status === "denied") && (
            <p className="font-sans text-sm text-ink mt-2 border-l-2 border-line pl-3">
              <span className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate block mb-1">
                Admin note
              </span>
              {dispute.resolutionNote}
            </p>
          )}
      </div>
    </div>
  );
}

function TimelineLine({
  label,
  timestamp,
  muted,
}: {
  label: string;
  timestamp: string | null;
  muted?: boolean;
}) {
  return (
    <li className={cn("flex flex-wrap gap-x-3", muted && "text-slate-soft")}>
      <span className={muted ? "text-slate-soft" : "text-ink-soft"}>
        {label}
      </span>
      {timestamp && (
        <span className="font-mono tnum text-slate">
          {formatTimelineDate(timestamp)}
        </span>
      )}
    </li>
  );
}

function formatTimelineDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  // Mon DD, YYYY · HH:MM — matches the Quiet Studio editorial date format
  // used elsewhere (formatLongDate + paidAt rendering).
  const date = d.toLocaleDateString("en-PH", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const time = d.toLocaleTimeString("en-PH", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  return `${date} · ${time}`;
}

const STATUS_BADGE: Record<
  RunnerDisputeStatus,
  { dot: string; text: string; label: string }
> = {
  open: { dot: "bg-slate", text: "text-slate", label: "Open" },
  escalated: { dot: "bg-slate", text: "text-slate", label: "In review" },
  resolved: { dot: "bg-fresh", text: "text-fresh", label: "Resolved" },
  denied: { dot: "bg-error", text: "text-error", label: "Denied" },
  withdrawn: { dot: "bg-slate-soft", text: "text-slate-soft", label: "Withdrawn" },
};
