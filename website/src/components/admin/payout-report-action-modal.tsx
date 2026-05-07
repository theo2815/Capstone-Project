"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import type { PayoutReport } from "@/lib/admin-payout-reports";

export type PayoutReportActionMode = "acknowledge" | "resolve";

interface PayoutReportActionModalProps {
  open: boolean;
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
    placeholder: "We're verifying with the bank — will resolve within 24h.",
    submit: "Send acknowledgement",
    minLength: 10,
  },
  resolve: {
    title: (id) => `Resolve · ${id}`,
    intro: (handle) =>
      handle
        ? `Closes the report and notifies @${handle}. Note should explain what happened and how it was fixed (e.g. transfer reference).`
        : "Closes the report and notifies the photographer. Note should explain what happened and how it was fixed (e.g. transfer reference).",
    label: "Resolution note (sent to photographer)",
    placeholder: "GCash retry succeeded · ref MY-32C887R · funds posted today.",
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
      onClose={onClose}
      onSubmit={handleSubmit}
      title={config.title(report.id)}
      intro={config.intro(report.handle)}
      submitLabel={config.submit}
      submitDisabled={!valid}
    >
      <AdminTextarea
        id={`report-action-${mode}-note`}
        label={config.label}
        value={note}
        onChange={setNote}
        maxLength={500}
        placeholder={config.placeholder}
        rows={5}
      />
    </AdminFormModal>
  );
}
