"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";

interface AdminEscalateModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (note: string | null) => void;
  /** Header line — e.g. "Escalate dispute DSP-A1B2C3" or "Escalate flag FLG-001" */
  targetLabel: string;
  /** Body copy describing what escalation does in this domain. */
  body: string;
}

// Shared escalation modal — used by both disputes (action aside) and flags
// (inline card). Optional note only; no canned reasons. Submit is enabled
// even without a note so admin can escalate quickly.
export function AdminEscalateModal({
  open,
  onClose,
  onSubmit,
  targetLabel,
  body,
}: AdminEscalateModalProps) {
  const [note, setNote] = useState("");

  function handleSubmit() {
    const trimmed = note.trim();
    onSubmit(trimmed.length > 0 ? trimmed : null);
    setNote("");
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={targetLabel}
      intro={body}
      submitLabel="Escalate"
    >
      <AdminTextarea
        id="escalate-note"
        label="Note for next reviewer (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="Why is this beyond your tier?"
      />
    </AdminFormModal>
  );
}
