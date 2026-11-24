"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminSuspendModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  /** Resolves false when the request failed, so the modal keeps its input and shows why. */
  onSubmit: (reason: string) => Promise<boolean | void>;
  photographerName: string;
  /** What suspending does to this account; defaults to the photographer wording. */
  intro?: string;
  /** The reasons to offer; defaults to the photographer list. The `other` code makes the note required. */
  reasons?: ReadonlyArray<{ value: string; label: string }>;
}

// Runner reasons, Theo's wording (2026-11-21). Only the label is sent; the codes are radio values.
export const RUNNER_SUSPEND_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "false_reports", label: "Abusive or false reports against photographers" },
  { value: "false_claims", label: "Repeated false refund or dispute claims" },
  { value: "payment_fraud", label: "Suspected payment fraud or chargebacks" },
  { value: "identity_misuse", label: "Using someone else's selfie or account" },
  { value: "compliance_review", label: "Compliance review requested" },
  { value: "other", label: "Other (see note)" },
];

const SUSPEND_REASONS: ReadonlyArray<{ value: string; label: string }> = [
  { value: "watermark_violation", label: "Watermark or branding violates policy" },
  { value: "fraud_suspected", label: "Suspected fraudulent uploads" },
  { value: "chargeback_pattern", label: "Many payment disputes from buyers" },
  { value: "complaint_volume", label: "Multiple unresolved runner complaints" },
  { value: "compliance_review", label: "Compliance review requested" },
  { value: "other", label: "Other (see note)" },
];

// Suspend-with-reason modal. Same shape as the verifications Reject modal —
// canned reasons + 280-char note. The composed string flows through
// useAdminUserActions().suspend(userId, reason). Suspension is reversible from
// the same action aside via the Unsuspend button.
export function AdminSuspendModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  photographerName,
  intro,
  reasons = SUSPEND_REASONS,
}: AdminSuspendModalProps) {
  // AX-28: no reason starts picked, so a quick Enter never records one the admin did not choose.
  const [selectedReason, setSelectedReason] = useState("");
  const [note, setNote] = useState("");
  const [attempted, setAttempted] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!open) { setNote(""); setSelectedReason(""); setAttempted(false); setFailed(false); }
  }, [open]);

  const noteMissing = selectedReason === "other" && note.trim().length === 0;

  async function handleSubmit() {
    if (submitting) return;
    if (noteMissing) { setAttempted(true); return; }
    const label =
      reasons.find((r) => r.value === selectedReason)?.label ??
      selectedReason;
    const composed =
      note.trim().length > 0 ? `${label} — ${note.trim()}` : label;
    setFailed(false);
    setFailed((await onSubmit(composed)) === false);
  }

  return (
    <AdminFormModal
      tone="danger"
      open={open}
      submitting={submitting}
      pendingLabel="Suspending…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Suspend account"
      intro={intro ?? `${photographerName} is signed out shortly and cannot sign in until you unsuspend. Their photos come off sale and their public page stops opening; photos already bought can still be downloaded. Payouts are not stopped: hold those on the Payouts page. They are not emailed and cannot sign in to read the reason, so tell them yourself if they should know.`}
      submitLabel="Suspend"
      submitDisabled={!selectedReason}
      submitDescribedBy={selectedReason ? undefined : "suspend-reason-needed"}
      error={failed ? "Couldn't finish this. Your reason and note are still here." : null}
    >
      <AdminFormSection label="Reason">
        <AdminRadioGroup
          name="suspend-reason"
          options={reasons}
          value={selectedReason}
          onChange={setSelectedReason}
        />
      </AdminFormSection>
      <AdminTextarea
        id="suspend-note"
        label={selectedReason === "other" ? "Note (required, max 280)" : "Note (optional, max 280)"}
        error={attempted && noteMissing ? "Add a note: Other needs one." : null}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="Add detail. They can read this once they are unsuspended."
        hint="Seen by: the suspended person, once they can sign in again."
      />
      {!selectedReason && <p id="suspend-reason-needed" className="font-sans text-sm text-slate">Choose a reason to continue.</p>}
    </AdminFormModal>
  );
}
