"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextInput } from "@/components/admin/admin-form-fields";
import { PayoutAccountCard } from "@/components/admin/payout-account-card";
import { payoutMethodLabel, type AdminPayoutCycle } from "@/lib/admin-payouts";
import { formatPrice } from "@/lib/utils";

interface AdminPayoutMarkPaidModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (reference: string) => void;
  cycle: AdminPayoutCycle;
}

// Mark-paid modal. Single-row only — bulk mark-paid would force one
// shared reference across cycles, which doesn't make sense (each transfer
// has its own GCash/Maya reference). Reference is required, non-empty,
// pattern-relaxed (any non-whitespace chars allowed).
//
// Renders the photographer's payout account snapshot at the top — full
// account number, name, method, and a downloadable QR code — so admin can
// transfer via banking app (typed digits) or QR scan, then come back and
// enter the reference.
export function AdminPayoutMarkPaidModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  cycle,
}: AdminPayoutMarkPaidModalProps) {
  const [reference, setReference] = useState("");

  useEffect(() => {
    if (!open) setReference("");
  }, [open]);

  const trimmed = reference.trim();
  const valid = trimmed.length >= 4;

  function handleSubmit() {
    if (!valid) return;
    onSubmit(trimmed);
  }

  return (
    <AdminFormModal
      open={open}
      submitting={submitting}
      pendingLabel="Marking paid…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Mark paid · ${cycle.id}`}
      intro={
        <>
          Only mark paid after{" "}
          <span className="text-ink tnum">{formatPrice(cycle.amount)}</span>{" "}
          has left your account by {payoutMethodLabel(cycle.method)}. The
          photographer is told at once, with this reference. After this, the
          only change you can make is to mark the payout returned.
        </>
      }
      submitLabel="Mark paid"
      submitDisabled={!valid}
    >
      <PayoutAccountCard cycle={cycle} mode="modal" />
      <AdminTextInput
        id="payment-reference"
        label="Payment reference (required)"
        value={reference}
        onChange={setReference}
        maxLength={32}
        placeholder="e.g. GC-77B221"
        autoFocus
        inputClassName="tnum"
        hint="At least 4 characters."
        hintTone="calm"
      />
    </AdminFormModal>
  );
}
