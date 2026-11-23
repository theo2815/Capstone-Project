"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import type { PayoutReport } from "@/lib/admin-payout-reports";

export type PayoutReportActionMode = "acknowledge" | "resolve";

interface PayoutReportActionModalProps {
  open: boolean;
  submitting?: boolean;
  mode: PayoutReportActionMode;
  report: PayoutReport;
  onClose: () => void;
  onSubmit: (note: string) => void;
}

const COPY: Record<
  PayoutReportActionMode,
  {
    title: (id: string) => string;
    intro: (handle: string | null) => string;
    label: string;
    placeholder: string;
    submit: string;
    minLength: number;
  }
> = {
  acknowledge: {
    title: (id) => `Acknowledge · ${id}`,
    intro: (handle) =>
      handle
        ? `Reply lands in @${handle}'s inbox. Use this to confirm the report was received and what you're investigating next.`
        : "Reply lands in the photographer's inbox. Use this to confirm the report was received and what you're investigating next.",
    label: "Reply (sent to photographer)",
    placeholder: "What you are checking, and when you will write again",
    submit: "Send acknowledgement",
    minLength: 10,
  },
  resolve: {
    title: (id) => `Resolve · ${id}`,
    intro: (handle) =>
      handle
        ? `Closes the report and sends your note to @${handle}. Say what happened and how it was fixed, with the transfer reference. This is final: a resolved report cannot be reopened.`
        : "Closes the report and sends your note to the photographer. Say what happened and how it was fixed, with the transfer reference. This is final: a resolved report cannot be reopened.",
    label: "Resolution note (sent to photographer)",
    placeholder: "What happened and how it was fixed, with the transfer reference",
    submit: "Resolve report",
    minLength: 10,
  },
};

// Dual-mode action modal for the photographer-payout-reports section.
// `mode = "acknowledge"` flips the report to acknowledged + sends an
// inbox message; `mode = "resolve"` closes the report + sends the final
// resolution note. Both share the same form shape (textarea + 500 max).
export function PayoutReportActionModal({
  open,
  submitting = false,
  mode,
  report,
  onClose,
  onSubmit,
}: PayoutReportActionModalProps) {
  const [note, setNote] = useState("");

  useEffect(() => {
    if (!open) setNote("");
  }, [open]);

  const config = COPY[mode];
  const trimmed = note.trim();
  const valid = trimmed.length >= config.minLength;

  function handleSubmit() {
    if (!valid) return;
    onSubmit(trimmed);
  }

  return (
    <AdminFormModal
      open={open}
      submitting={submitting}
      pendingLabel={mode === "acknowledge" ? "Acknowledging…" : "Resolving…"}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={config.title(report.id)}
      intro={config.intro(report.handle)}
      submitLabel={config.submit}
      submitDisabled={!valid}
    >
      <section className="space-y-2">
        <AdminTextarea
          id={`report-action-${mode}-note`}
          label={config.label}
          value={note}
          onChange={setNote}
          maxLength={500}
          placeholder={config.placeholder}
          rows={5}
          describedBy={`report-action-${mode}-note-hint`}
        />
        <p id={`report-action-${mode}-note-hint`} className="font-sans text-sm text-slate-soft">
          At least 10 characters.
        </p>
      </section>
    </AdminFormModal>
  );
}
