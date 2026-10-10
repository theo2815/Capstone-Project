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
import { usePayoutReportActions } from "@/hooks/use-payout-reports";
import { useToast } from "@/hooks/use-toast";
import { resolveCurrentPhotographer } from "@/lib/current-photographer";
import { useAuth } from "@/hooks/use-auth";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
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
// optional (max 500). Submits to the backend; admin sees
// the new row in /admin/payouts under "Reports from photographers".
export function FilePayoutReportModal({
  cycle,
  onClose,
}: FilePayoutReportModalProps) {
  const { user } = useAuth();
  const { submit } = usePayoutReportActions();
  const { showToast } = useToast();

  const [reason, setReason] = useState<PayoutReportReason | null>(null);
  const [reasonOpen, setReasonOpen] = useState(false);
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

  const settingsHandle = usePhotographerSettingsStore((s) => s.handle);
  const photographer = resolveCurrentPhotographer(user, settingsHandle);

  if (!cycle) return null;

  async function handleSubmit() {
    if (!cycle || submitting) return;
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
    const ok = await submit({ payoutId: cycle.id, reason, note: note.trim() || null });
    setSubmitting(false);
    if (!ok) return;
    showToast({
      kind: "success",
      message: "Report sent. Admin will review and reply in your inbox.",
    });
    onClose();
  }

  const reasonLabel = reason ? PAYOUT_REPORT_REASON_LABEL[reason] : "Pick a reason";

  return (
    <Modal isOpen={cycle !== null} onClose={() => { if (!submitting) onClose(); }} hideClose={submitting} title="File a report" escDisabled={reasonOpen}>
      <Kicker as="p" tone="soft" tnum className="mb-5">
        PAYOUT · {cycle.id} · {formatLongDate(cycle.requestedAt)} ·{" "}
        {formatPrice(cycle.amount)}
      </Kicker>

      <fieldset disabled={submitting} className="min-w-0 space-y-5">
        <div>
          <Kicker as="label" tone="soft" className="mb-2 block">
            Reason
          </Kicker>
          <Dropdown
            ariaLabel="Report reason"
            panelClassName="z-[70]"
            onOpenChange={setReasonOpen}
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
                  if (submitting) return;
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
      </fieldset>

      <div className="mt-6 flex items-center justify-end gap-3">
        <button
          type="button"
          onClick={onClose}
          disabled={submitting}
          className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-4 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={submitting || !reason}
          className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink border border-line hover:bg-ink hover:text-surface hover:border-ink transition-colors rounded-full px-5 py-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? "Sending…" : "Send report"}
        </button>
      </div>
    </Modal>
  );
}
