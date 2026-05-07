"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextInput,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";
import type { DisputeResolution } from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";

interface AdminResolveDisputeModalProps {
  open: boolean;
  /** When true, parent disables ESC handling (stacked refund-confirm child is open). */
  escDisabled?: boolean;
  onClose: () => void;
  /** Continue to refund confirmation — args reflect the form's current state. */
  onContinue: (args: {
    resolution: DisputeResolution;
    refundAmount: number | null;
    note: string;
  }) => void;
  disputeId: string;
  orderTotal: number;
}

// Resolve modal for /admin/disputes/[id]. Carries the resolution radio +
// refund-amount input (visible only on Partial) + note. On submit it does
// NOT mutate the store directly — the parent action-aside opens a stacked
// `<AdminRefundConfirmModal>` first. State stays here and is lifted to
// the parent on Continue so the confirm-modal can read the same values.
export function AdminResolveDisputeModal({
  open,
  escDisabled = false,
  onClose,
  onContinue,
  disputeId,
  orderTotal,
}: AdminResolveDisputeModalProps) {
  const [resolution, setResolution] = useState<DisputeResolution>("refund_full");
  const [partialAmount, setPartialAmount] = useState<string>("");
  const [note, setNote] = useState("");

  // Reset whenever the modal closes so the next open starts fresh.
  useEffect(() => {
    if (!open) {
      setResolution("refund_full");
      setPartialAmount("");
      setNote("");
    }
  }, [open]);

  const parsedPartial = Number(partialAmount);
  const partialValid =
    Number.isFinite(parsedPartial) &&
    parsedPartial > 0 &&
    parsedPartial <= orderTotal;

  const continueDisabled =
    note.trim().length === 0 ||
    (resolution === "refund_partial" && !partialValid);

  const partialHint =
    resolution === "refund_partial" && partialAmount.length > 0 && !partialValid
      ? (
          <>
            Must be between 1 and{" "}
            <span className="tnum">{orderTotal}</span>.
          </>
        )
      : undefined;

  function handleSubmit() {
    const amount =
      resolution === "refund_full"
        ? orderTotal
        : resolution === "refund_partial"
          ? parsedPartial
          : null;
    onContinue({ resolution, refundAmount: amount, note: note.trim() });
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Resolve ${disputeId}`}
      escDisabled={escDisabled}
      intro={
        <>
          Order total{" "}
          <span className="text-ink tnum">{formatPrice(orderTotal)}</span>.
          Choose how to settle this dispute.
        </>
      }
      submitLabel={resolution === "deny" ? "Deny" : "Continue"}
      submitDisabled={continueDisabled}
    >
      <AdminFormSection label="Resolution">
        <AdminRadioGroup
          name="resolution"
          value={resolution}
          onChange={(v) => setResolution(v as DisputeResolution)}
          options={[
            {
              value: "refund_full",
              label: (
                <>
                  Full refund —{" "}
                  <span className="text-ink tnum">{formatPrice(orderTotal)}</span>{" "}
                  returned to runner.
                </>
              ),
            },
            {
              value: "refund_partial",
              label: <>Partial refund — enter amount below.</>,
            },
            {
              value: "deny",
              label: <>Deny — no refund, dispute closes.</>,
            },
          ]}
        />
      </AdminFormSection>

      {resolution === "refund_partial" && (
        <AdminTextInput
          id="partial-amount"
          label={
            <>
              Refund amount (₱, max <span className="tnum">{orderTotal}</span>)
            </>
          }
          type="number"
          inputMode="decimal"
          step="1"
          min={1}
          max={orderTotal}
          value={partialAmount}
          onChange={setPartialAmount}
          placeholder="e.g. 100"
          inputClassName="tnum"
          hint={partialHint}
        />
      )}

      <AdminTextarea
        id="resolve-note"
        label="Reasoning (required, max 600)"
        value={note}
        onChange={setNote}
        maxLength={600}
        rows={4}
        placeholder="What did you decide and why?"
      />
    </AdminFormModal>
  );
}
