"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminDenyDisputeModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (reason: string) => void;
  disputeId: string;
}

const DENY_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "outside_window", label: "Outside refund window (>30 days)" },
  { value: "evidence_insufficient", label: "Evidence doesn't support claim" },
  { value: "policy_excluded", label: "Reason excluded by platform policy" },
  { value: "duplicate_claim", label: "Duplicate of an earlier dispute" },
  { value: "runner_resolved", label: "Runner already resolved with photographer" },
  { value: "other", label: "Other (see note)" },
];

// Deny-with-reason modal for /admin/disputes/[id]. Same shape as the
// verifications reject modal — radio + 280-char note composed into the
// reason string passed to useAdminDisputeStore.deny(disputeId, reason).
export function AdminDenyDisputeModal({
  open,
  onClose,
  onSubmit,
  disputeId,
}: AdminDenyDisputeModalProps) {
  const [selectedReason, setSelectedReason] = useState<string>(
    DENY_REASONS[0]!.value,
  );
  const [note, setNote] = useState("");

  function handleSubmit() {
    const label =
      DENY_REASONS.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    onSubmit(composed);
    setNote("");
    setSelectedReason(DENY_REASONS[0]!.value);
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Deny ${disputeId}`}
      intro="The runner sees the reason in their orders ledger. No refund is issued. They can re-file with new evidence within 7 days."
      submitLabel="Deny claim"
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="dispute-deny-reason"
          options={DENY_REASONS}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="deny-note"
        label="Note (optional, max 280)"
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the runner see?"
      />
    </AdminFormModal>
  );
}
