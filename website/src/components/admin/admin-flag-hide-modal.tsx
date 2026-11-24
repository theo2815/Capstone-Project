"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminFlagHideModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (note: string | null) => void;
  flagId: string;
}

const HIDE_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "policy_violation", label: "Violates platform policy" },
  { value: "wrong_runner_confirmed", label: "Wrong-runner match confirmed" },
  { value: "quality_below_threshold", label: "Quality below platform threshold" },
  { value: "watermark_compromised", label: "Watermark integrity compromised" },
  { value: "duplicate_listing", label: "Duplicate of another listing" },
  { value: "other", label: "Other (see note)" },
];

// Hide-flag confirmation modal. Same shape as suspend/reject — radio +
// 280-char note. Submission marks the flag hidden and the photo will not
// appear in runner-facing search results going forward (Phase F backend).
export function AdminFlagHideModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  flagId,
}: AdminFlagHideModalProps) {
  // C-67 (as AX-28 did for the other reason dialogs): no reason starts picked, so a quick Enter never hides with one.
  const [selectedReason, setSelectedReason] = useState("");
  const [note, setNote] = useState("");
  const [attempted, setAttempted] = useState(false);
  const noteMissing = selectedReason === "other" && note.trim().length === 0;

  useEffect(() => {
    if (!open) { setNote(""); setSelectedReason(""); setAttempted(false); }
  }, [open]);

  function handleSubmit() {
    if (noteMissing) { setAttempted(true); return; }
    const label =
      HIDE_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);

  }

  return (
    <AdminFormModal
      tone="danger"
      open={open}
      submitting={submitting}
      pendingLabel="Hiding…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Hide ${flagId}`}
      intro="The photo is hidden from search and galleries. Nobody is told automatically: the reason is saved on the flag for admins. To tell the photographer, send them a message from their page."
      submitLabel="Hide photo"
      submitDisabled={!selectedReason}
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="hide-reason"
          options={HIDE_REASONS}
          value={selectedReason}
          onChange={(reason) => { setSelectedReason(reason); setAttempted(false); }}
        />
      </AdminFormSection>
      <AdminTextarea
        id="hide-note"
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        hint="Seen by: admins only."
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="Anything another admin should know"
      />
      <p className="font-sans text-sm text-ink-soft">To undo, dismiss or resolve this flag: the photo goes back on sale unless another flag still hides it.</p>
      {!selectedReason && <p className="font-sans text-sm text-slate">Choose a reason to continue.</p>}
    </AdminFormModal>
  );
}
