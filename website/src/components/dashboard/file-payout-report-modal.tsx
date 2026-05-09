"use client";

import { useEffect, useState } from "react";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import {
  Dropdown,
  DropdownItem,
  DropdownTrigger,
} from "@/components/ui/dropdown";
import { FieldError } from "@/components/ui/field-error";
import { CharCounter } from "@/components/ui/char-counter";
import {
  PAYOUT_REPORT_REASON_LABEL,
  type PayoutReportReason,
} from "@/lib/admin-payout-reports";
import { useAdminPayoutReportStore } from "@/store/admin-payout-report-store";
import { useToast } from "@/hooks/use-toast";
import { submitPayoutReport } from "@/lib/api-photographer-earnings";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { resolveCurrentPhotographer } from "@/lib/current-photographer";
import { useAuth } from "@/hooks/use-auth";
import type { PhotographerPayout } from "@/lib/photographer-mock";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";
import { cn } from "@/lib/utils";

interface FilePayoutReportModalProps {
  cycle: PhotographerPayout | null;
  onClose: () => void;
}

const NOTE_MAX = 500;
const REASON_OPTIONS = Object.entries(PAYOUT_REPORT_REASON_LABEL) as ReadonlyArray<
  [PayoutReportReason, string]
>;

// Photographer-side report form. Anchored to a specific PhotographerPayout
// row from /dashboard/billing — cycle context is read-only at the top of
// the modal, no cycle picker. Reason dropdown is required, note is
// optional (max 500). Submits into admin-payout-report-store; admin sees
// the new row in /admin/payouts under "Reports from photographers".
export function FilePayoutReportModal({
  cycle,
  onClose,
}: FilePayoutReportModalProps) {
  const { user } = useAuth();
  const submitReport = useAdminPayoutReportStore((s) => s.submitReport);
  const { showToast } = useToast();

  const [reason, setReason] = useState<PayoutReportReason | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (cycle === null) {
      setReason(null);
      setNote("");
      setError(null);
      setSubmitting(false);
    }
  }, [cycle]);

  const photographer = resolveCurrentPhotographer(user);

  if (!cycle) return null;

  function handleSubmit() {
    if (!cycle) return;
    if (!reason) {
      setError("Pick a reason for the report.");
      return;
    }
    if (!photographer) {
      setError("You must be signed in as a photographer to file a report.");
      return;
    }
    if (note.length > NOTE_MAX) {
      setError(`Note must be ${NOTE_MAX} characters or fewer.`);
      return;
    }
    setError(null);
    setSubmitting(true);
    // Local store update fires immediately for the UI snap; live mode also
    // POSTs the report to the backend (Q-A1 cycle ID). 600ms latency seam
    // mirrors the mock submit pacing so toasts don't feel rushed.
    setTimeout(() => {
      submitReport({
        payoutCycleId: cycle.id,
        photographerId: photographer.id,
        photographerName: photographer.name,
        handle: photographer.handle,
        reason,
        note: note.trim(),
      });
      if (BACKEND_LIVE) {
        void submitPayoutReport({
          payoutId: cycle.id,
          reason,
          note: note.trim() || null,
        }).catch((err) => {
          console.error("[photographer/payout-report] submit failed", err);
        });
      }
      setSubmitting(false);
      showToast({
        kind: "success",
        message: "Report sent. Admin will review and reply in your inbox.",
      });
      onClose();
    }, 600);
  }

  const reasonLabel = reason ? PAYOUT_REPORT_REASON_LABEL[reason] : "Pick a reason";

  return (
    <Modal isOpen={cycle !== null} onClose={onClose} title="File a report">
      <Kicker as="p" tone="soft" tnum className="mb-5">
        CYCLE · {cycle.id} · {formatLongDate(cycle.weekOf, true)} ·{" "}
        {formatPrice(cycle.amount)}
      </Kicker>

      <div className="space-y-5">
        <div>
          <Kicker as="label" tone="soft" className="mb-2 block">
            Reason
          </Kicker>
          <Dropdown
            ariaLabel="Report reason"
            panelClassName="z-[70]"
            trigger={
              <DropdownTrigger
                className={cn(
                  "border border-line rounded-full px-4 py-2 hover:border-ink transition-colors",
                  reason ? "text-ink" : "text-slate-soft",
                )}
              >
                {reasonLabel}
              </DropdownTrigger>
            }
          >
            {REASON_OPTIONS.map(([value, label]) => (
              <DropdownItem
                key={value}
                active={reason === value}
                onClick={() => {
                  setReason(value);
                  setError(null);
                }}
              >
                {label}
              </DropdownItem>
            ))}
          </Dropdown>
        </div>

        <div>
          <Kicker
            as="label"
            tone="soft"
            className="mb-2 block"
            htmlFor="report-note"
          >
            Note (optional)
          </Kicker>
          <textarea
            id="report-note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            maxLength={NOTE_MAX}
            placeholder="What happened? Reference numbers, expected vs received, etc."
            className="w-full font-sans text-sm text-ink bg-bone border border-line rounded-xl px-4 py-3 min-h-[120px] resize-y focus:outline-none focus:border-ink placeholder:text-slate-soft"
            aria-describedby="report-note-counter"
          />
          <CharCounter
            id="report-note-counter"
            current={note.length}
            max={NOTE_MAX}
            className="mt-1.5"
          />
        </div>

        <FieldError id="report-error" message={error ?? undefined} />
      </div>

      <div className="mt-6 flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onClose}
          disabled={submitting}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-4 py-2 disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={submitting || !reason}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? "Sending…" : "Send report"}
        </button>
      </div>
    </Modal>
  );
}
