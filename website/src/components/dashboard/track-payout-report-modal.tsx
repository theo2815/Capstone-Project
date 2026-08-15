"use client";

import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import {
  PAYOUT_REPORT_REASON_LABEL,
  PAYOUT_REPORT_STATUS_LABEL,
  type PayoutReport,
} from "@/lib/admin-payout-reports";
import type { PhotographerPayout } from "@/lib/photographer-mock";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";
import { cn } from "@/lib/utils";

// Read-only mirror of the admin Reports drawer, scoped to the photographer's
// own report on a single cycle. Replaces the "File a report" action on a
// cycle row whenever a report exists — covers all three states (open,
// acknowledged, resolved). When the latest report is resolved, the modal
// surfaces a "File a follow-up report →" link that closes the track view
// and re-opens the file-report modal on the same cycle, so the row stays
// at one button slot regardless of state.

interface TrackPayoutReportModalProps {
  /** The report being tracked. null closes the modal. */
  report: PayoutReport | null;
  /** The cycle the report belongs to — used for the context line. */
  cycle: PhotographerPayout | null;
  onClose: () => void;
  /**
   * Called when the photographer wants to file a follow-up report (only
   * surfaced when the current report is resolved). Caller closes this modal
   * and opens the file-report modal pointed at `cycle`.
   */
  onFileFollowUp?: () => void;
}

export function TrackPayoutReportModal({
  report,
  cycle,
  onClose,
  onFileFollowUp,
}: TrackPayoutReportModalProps) {
  if (!report || !cycle) return null;

  const statusTone =
    report.status === "open"
      ? "amber"
      : report.status === "acknowledged"
        ? "ink"
        : "fresh";

  return (
    <Modal isOpen={report !== null} onClose={onClose} title="Track your report">
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <AdminStatusPill
          label={PAYOUT_REPORT_STATUS_LABEL[report.status]}
          tone={statusTone}
        />
        <Kicker as="p" tone="soft" tnum>
          {report.id}
        </Kicker>
      </div>

      <Kicker as="p" tone="soft" tnum className="mb-6">
        PAYOUT · {cycle.id} · {formatLongDate(cycle.requestedAt)} ·{" "}
        {formatPrice(cycle.amount)}
      </Kicker>

      <section className="mb-6">
        <Kicker as="p" tone="soft" className="mb-3">
          Your submission
        </Kicker>
        <p className="font-sans text-base text-ink">
          {PAYOUT_REPORT_REASON_LABEL[report.reason]}
        </p>
        {report.note && (
          <p className="font-sans text-sm text-ink-soft mt-2 leading-relaxed">
            {report.note}
          </p>
        )}
      </section>

      <section className="mb-6">
        <Kicker as="p" tone="soft" className="mb-3">
          Admin response
        </Kicker>
        <AdminResponse report={report} />
      </section>

      <section className="mb-6">
        <Kicker as="p" tone="soft" className="mb-3">
          Timeline
        </Kicker>
        <ol className="space-y-3">
          <TimelineRow
            label="Filed"
            at={report.reportedAt}
            note="By you"
            active
          />
          <TimelineRow
            label="Acknowledged"
            at={report.acknowledgedAt}
            note={report.acknowledgeReply}
            active={report.status !== "open"}
          />
          <TimelineRow
            label="Resolved"
            at={report.resolvedAt}
            note={report.resolutionNote}
            active={report.status === "resolved"}
          />
        </ol>
      </section>

      <div className="mt-6 flex items-center justify-between gap-3 flex-wrap">
        {report.status === "resolved" && onFileFollowUp ? (
          <button
            type="button"
            onClick={onFileFollowUp}
            className="font-mono uppercase tracking-[0.22em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink"
          >
            File a follow-up report →
          </button>
        ) : (
          <span aria-hidden="true" />
        )}
        <button
          type="button"
          onClick={onClose}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2"
        >
          Close
        </button>
      </div>
    </Modal>
  );
}

function AdminResponse({ report }: { report: PayoutReport }) {
  if (report.status === "open") {
    return (
      <p className="font-sans text-sm text-slate-soft leading-relaxed">
        Awaiting admin reply. We&apos;ll send an inbox message as soon as they
        respond. Most reports get a first response within 24 hours.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {report.acknowledgeReply && (
        <div>
          <Kicker tone="soft" tnum className="mb-1.5">
            Acknowledged · {formatLongDate(report.acknowledgedAt ?? "")}
          </Kicker>
          <p className="font-sans text-sm text-ink leading-relaxed">
            {report.acknowledgeReply}
          </p>
        </div>
      )}
      {report.status === "resolved" && report.resolutionNote && (
        <div>
          <Kicker tone="soft" tnum className="mb-1.5">
            Resolved · {formatLongDate(report.resolvedAt ?? "")}
          </Kicker>
          <p className="font-sans text-sm text-ink leading-relaxed">
            {report.resolutionNote}
          </p>
        </div>
      )}
    </div>
  );
}

interface TimelineRowProps {
  label: string;
  at: string | null;
  note: string | null;
  active: boolean;
}

function TimelineRow({ label, at, note, active }: TimelineRowProps) {
  return (
    <li
      className={cn(
        "rounded-xl border px-4 py-3",
        active ? "border-line bg-bone-deep" : "border-line/60 bg-bone",
      )}
    >
      <div className="flex items-baseline justify-between gap-3">
        <Kicker tone={active ? "default" : "soft"}>{label}</Kicker>
        {at && (
          <Kicker as="time" tone="soft" tnum>
            {formatLongDate(at)}
          </Kicker>
        )}
      </div>
      {note ? (
        <p
          className={cn(
            "font-sans text-sm mt-2 leading-relaxed",
            active ? "text-ink-soft" : "text-slate-soft",
          )}
        >
          {note}
        </p>
      ) : !at ? (
        <p className="font-sans text-sm text-slate-soft mt-2">Pending.</p>
      ) : null}
    </li>
  );
}
