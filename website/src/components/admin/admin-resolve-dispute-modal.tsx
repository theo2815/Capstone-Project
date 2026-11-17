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
  submitting?: boolean;
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
  itemChargeAmount?: number | null;
  /** QR Ph can only refund the whole payment, so it offers a full refund and nothing else. */
  paymentMethod?: string;
  /** DW-3: false means a full refund cannot be approved now; Approve is then disabled. */
  refundable?: boolean | null;
  refundBlockedReason?: "bought_together" | "window_closed" | "no_payment" | null;
  refundBlockedMessage?: string | null;
  paymentPhotoCount?: number | null;
  /** The runner's reason, used to start the note ("Approved: <reason>."). */
  reasonLabel?: string;
  /** Swap to the Deny modal with this reason picked (bought_together / window_closed only). */
  onDenyWithReason?: (code: "bought_together" | "window_closed") => void;
}

// Resolve modal for /admin/disputes/[id]. Carries the resolution radio +
// refund-amount input (visible only on Partial) + note. On submit it does
// NOT mutate the store directly — the parent action-aside opens a stacked
// `<AdminRefundConfirmModal>` first. State stays here and is lifted to
// the parent on Continue so the confirm-modal can read the same values.
export function AdminResolveDisputeModal({
  open,
  submitting = false,
  escDisabled = false,
  onClose,
  onContinue,
  disputeId,
  orderTotal,
  itemChargeAmount,
  paymentMethod,
  refundable = null,
  refundBlockedReason = null,
  refundBlockedMessage = null,
  paymentPhotoCount = null,
  reasonLabel,
  onDenyWithReason,
}: AdminResolveDisputeModalProps) {
  const [resolution, setResolution] = useState<DisputeResolution>("refund_full");
  const [partialAmount, setPartialAmount] = useState<string>("");
  const [note, setNote] = useState("");

  const prefill = reasonLabel ? `Approved: ${reasonLabel}.` : "";
  // Start filled in on open; reset whenever the modal closes so the next open starts fresh.
  useEffect(() => {
    setResolution("refund_full");
    setPartialAmount("");
    setNote(open ? prefill : "");
  }, [open, prefill]);

  const fullOnly = paymentMethod === "qrph";
  const blocked = refundable === false;
  const denyCode = refundBlockedReason === "bought_together" || refundBlockedReason === "window_closed" ? refundBlockedReason : null;

  const parsedPartial = Number(partialAmount);
  const chargeAvailable = itemChargeAmount != null && Number.isFinite(itemChargeAmount) && itemChargeAmount > 0;
  const partialValid =
    chargeAvailable &&
    /^\d+(?:\.\d{1,2})?$/.test(partialAmount) &&
    Number.isFinite(parsedPartial) &&
    parsedPartial > 0 &&
    parsedPartial < itemChargeAmount;

  const continueDisabled =
    blocked ||
    note.trim().length === 0 ||
    !chargeAvailable ||
    (resolution === "refund_partial" && !partialValid);

  const partialHint =
    resolution === "refund_partial" && partialAmount.length > 0 && !partialValid
      ? (
          <>
            Enter an amount above zero and below{" "}
            <span className="tnum">{formatPrice(itemChargeAmount ?? 0)}</span>, with up to two decimals.
          </>
        )
      : undefined;

  function handleSubmit() {
    const amount =
      resolution === "refund_full"
        ? itemChargeAmount ?? null
        : resolution === "refund_partial"
          ? parsedPartial
          : null;
    onContinue({ resolution, refundAmount: amount, note: note.trim() });
  }

  return (
    <AdminFormModal
      open={open}
      submitting={submitting}
      pendingLabel="Resolving…"
      onClose={onClose}
      onSubmit={handleSubmit}
      title={`Resolve ${disputeId}`}
      escDisabled={escDisabled}
      intro={
        <>
          {chargeAvailable && (
            <>
              This photo&apos;s charge <span className="text-ink tnum">{formatPrice(itemChargeAmount)}</span>
              {" "}(order total <span className="tnum">{formatPrice(orderTotal)}</span>).
            </>
          )}
          {chargeAvailable
            ? " Choose how to settle this dispute."
            : itemChargeAmount === 0
              ? <>Order total <span className="tnum">{formatPrice(orderTotal)}</span>. This item has no refundable charge.</>
              : <>Order total <span className="tnum">{formatPrice(orderTotal)}</span>. Item charge unavailable; retry when order details load.</>}
        </>
      }
      submitLabel="Continue"
      submitDisabled={continueDisabled}
    >
      {blocked && (
        <div role="alert" className="space-y-2 rounded-lg border border-line p-3 font-sans text-sm text-ink-soft">
          <p>
            {refundBlockedMessage ?? "A refund can't be approved for this dispute."}
            {refundBlockedReason === "bought_together" && paymentPhotoCount != null && (
              <> The payment covers <span className="tnum">{paymentPhotoCount}</span> photos.</>
            )}
          </p>
          {denyCode && onDenyWithReason && (
            <button
              type="button"
              onClick={() => onDenyWithReason(denyCode)}
              className="font-mono uppercase tracking-[0.14em] text-[13px] text-ink underline"
            >
              Deny with this reason
            </button>
          )}
        </div>
      )}
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
                  {chargeAvailable
                    ? <><span className="text-ink tnum">{formatPrice(itemChargeAmount)}</span> returned to runner.</>
                    : itemChargeAmount === 0 ? "no refundable charge." : "amount unavailable."}
                </>
              ),
            },
            ...(fullOnly
              ? []
              : [{ value: "refund_partial", label: <>Partial refund — enter amount below.</> }]),
          ]}
        />
      </AdminFormSection>

      {resolution === "refund_partial" && (
        <AdminTextInput
          id="partial-amount"
          label={
            <>
              Refund amount (₱, below <span className="tnum">{formatPrice(itemChargeAmount ?? 0)}</span>)
            </>
          }
          type="number"
          inputMode="decimal"
          step="0.01"
          min={0.01}
          max={itemChargeAmount ?? undefined}
          value={partialAmount}
          onChange={setPartialAmount}
          placeholder="e.g. 100"
          inputClassName="tnum"
          hint={partialHint}
        />
      )}

      <AdminTextarea
        id="resolve-note"
        label="Reasoning (required, max 500)"
        value={note}
        onChange={setNote}
        maxLength={500}
        rows={4}
        placeholder="What did you decide and why?"
      />
    </AdminFormModal>
  );
}
