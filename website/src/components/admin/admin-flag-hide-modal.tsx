"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminFlagHideModalProps {
  open: boolean;
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
  onClose,
  onSubmit,
  flagId,
}: AdminFlagHideModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    HIDE_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  function handleSubmit() {
    const label =
      HIDE_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(HIDE_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Hide ${flagId}`}
      intro="The photo will be hidden from runner-facing search and gallery views. The photographer is notified with the reason and can appeal via support."
      submitLabel="Hide photo"
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="hide-reason"
          options={HIDE_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="hide-note"
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the photographer see?"
      />
    </AdminFormModal>
  );
}
