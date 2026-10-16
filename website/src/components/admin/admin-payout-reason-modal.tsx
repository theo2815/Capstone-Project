"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { formatPrice } from "@/lib/utils";

// One reason-capturing modal for the three payout transitions that require a
// note: reject, mark-failed, mark-returned. Free-text (not the preset radios
// the hold modal uses) — these reasons are one-off explanations, not a fixed
// taxonomy. The reason is required (submit stays disabled until non-empty) and
// capped at 255 to match PayoutReasonRequest on the backend.

export type PayoutReasonKind = "reject" | "failed" | "returned";

const COPY: Record<
  PayoutReasonKind,
  { title: string; verb: string; submit: string; placeholder: string; note: string }
> = {
  reject: {
    title: "Reject payout",
    verb: "rejected",
    submit: "Reject payout",
    placeholder: "Why is this request being rejected?",
    note: "The held funds return to the photographer's balance and they're notified with this reason.",
  },
  failed: {
    title: "Mark payout failed",
    verb: "failed",
    submit: "Mark failed",
    placeholder: "What went wrong with the transfer?",
    note: "Use this when the money never left — the funds return to the photographer's balance to retry.",
  },
  returned: {
    title: "Mark payout returned",
    verb: "returned",
    submit: "Mark returned",
    placeholder: "Why did the transfer bounce back?",
    note: "Use this when the money left but came back — the photographer is notified to check their account.",
  },
};

export function AdminPayoutReasonModal({
  open,
  submitting = false,
  kind,
  payoutId,
  amount,
  onClose,
  onSubmit,
}: {
  open: boolean;
  submitting?: boolean;
  kind: PayoutReasonKind;
  payoutId: string;
  amount: number;
  onClose: () => void;
  onSubmit: (reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const copy = COPY[kind];
  const trimmed = reason.trim();

  function close() {
    setReason("");
    onClose();
  }

  function handleSubmit() {
    if (trimmed.length === 0) return;
    onSubmit(trimmed);
  }

  return (
    <AdminFormModal
      open={open}
      submitting={submitting}
      pendingLabel={kind === "reject" ? "Rejecting…" : kind === "failed" ? "Marking failed…" : "Marking returned…"}
      onClose={close}
      onSubmit={handleSubmit}
      title={`${copy.title} · ${payoutId}`}
      intro={
        <>
          This cycle{" "}
          <span className="text-ink tnum">({formatPrice(amount)})</span> will be
          marked {copy.verb}. {copy.note}
        </>
      }
      submitLabel={copy.submit}
      submitDisabled={trimmed.length === 0}
    >
      <AdminTextarea
        id="payout-reason"
        label="Reason (max 255)"
        value={reason}
        onChange={setReason}
        maxLength={255}
        placeholder={copy.placeholder}
      />
    </AdminFormModal>
  );
}
