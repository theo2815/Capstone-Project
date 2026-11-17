"use client";

import { useLayoutEffect, useState } from "react";
import { useDisputeActions } from "@/hooks/use-admin-disputes";
import { useToast } from "@/hooks/use-toast";
import { DISPUTE_REASON_LABEL, type Dispute, type DisputeResolution } from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";
import { AdminResolveDisputeModal } from "./admin-resolve-dispute-modal";
import { AdminRefundConfirmModal } from "./admin-refund-confirm-modal";
import { AdminDenyDisputeModal } from "./admin-deny-dispute-modal";
import { AdminEscalateModal } from "./admin-escalate-modal";
import { AdminRefundPanel } from "./admin-refund-panel";
import { Kicker } from "@/components/ui/kicker";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";

interface AdminDisputeActionAsideProps {
  dispute: Dispute;
  runnerHandle: string;
  // Reports whether any of the aside's dialogs is open, so a host drawer can stand its Esc down.
  onDialogOpenChange?: (open: boolean) => void;
  // Replaces the page-column layout (sticky on lg); the card look stays.
  className?: string;
}

interface PendingResolve {
  resolution: DisputeResolution;
  refundAmount: number | null;
  note: string;
}

// Sticky-on-lg / inline-on-md action cluster for the dispute detail
// page. Carries Approve refund / Deny / Escalate. Approve uses a stacked-modal
// pattern: open <AdminResolveDisputeModal>, capture form values into
// `pendingResolve`, then mount <AdminRefundConfirmModal> on top with the
// parent in escDisabled mode. Cancel on the child returns to the parent
// with state intact (lift state to this aside, not the modal).
//
// One-fresh-per-viewport: only Resolve is bg-fresh in its rest state on
// the dispute detail page. Deny carries the shared BTN_DANGER recipe so the
// destructive branch reads as consequential before its modal opens.
export function AdminDisputeActionAside({
  dispute,
  runnerHandle,
  onDialogOpenChange,
  className = "lg:sticky lg:top-24 mt-12 lg:mt-0",
}: AdminDisputeActionAsideProps) {
  const { showToast } = useToast();
  const { resolve, denyWithCode, escalate, isPending, pendingAction } = useDisputeActions();

  const submitting = isPending(dispute.id);
  const [resolveOpen, setResolveOpen] = useState(false);
  const [denyOpen, setDenyOpen] = useState(false);
  // The reason the Deny modal opens with, set by "Deny with this reason" in the Approve modal.
  const [denyCode, setDenyCode] = useState<string | null>(null);
  const [escalateOpen, setEscalateOpen] = useState(false);

  // Lifted state for the stacked refund-confirm flow.
  const [pendingResolve, setPendingResolve] = useState<PendingResolve | null>(
    null,
  );
  // The refund panel's own confirm and cancel dialogs.
  const [panelDialogOpen, setPanelDialogOpen] = useState(false);
  const dialogOpen = resolveOpen || denyOpen || escalateOpen || pendingResolve !== null || panelDialogOpen;
  // One writer, so a late "closed" from one modal can never clear another's "open".
  // The cleanup also clears the host flag if the aside unmounts with a dialog open.
  // Layout effect: the host hears it in the same commit, so no Esc lands in between.
  useLayoutEffect(() => {
    onDialogOpenChange?.(dialogOpen);
    return () => onDialogOpenChange?.(false);
  }, [dialogOpen, onDialogOpenChange]);

  // An escalated dispute is still the admin's to decide; only Escalate is for open ones.
  // (An approved refund that is still parked also has status "escalated", but its attention differs.)
  const isEscalated = dispute.attention === "escalated";
  const isOpen = dispute.status === "open" || isEscalated;

  function handleContinueResolve(args: PendingResolve) {
    if (submitting) return;
    setPendingResolve(args);
  }

  // A swap, not a stack: one state commit closes Approve and opens Deny, so the host never sees "no dialog".
  function handleSwapToDeny(code: string) {
    setPendingResolve(null);
    setResolveOpen(false);
    setDenyCode(code);
    setDenyOpen(true);
  }

  async function handleConfirmRefund() {
    if (!pendingResolve || submitting) return;
    const confirmed = await resolve(dispute.id, {
      resolution: pendingResolve.resolution,
      refundAmount: pendingResolve.refundAmount,
      reason: pendingResolve.note,
    });
    if (!confirmed) return;
    setPendingResolve(null);
    setResolveOpen(false);
    const refunded = confirmed.status === "resolved" && confirmed.refundAmount !== null;
    showToast({
      kind: refunded ? "success" : "info",
      message: refunded
        ? `Refunded · ${confirmed.id} · ${formatPrice(confirmed.refundAmount!)}`
        : `Refund requested · ${confirmed.id}`,
    });
  }

  function handleCancelRefund() {
    // Back to parent — pendingResolve cleared, parent state still set.
    setPendingResolve(null);
  }

  async function handleDeny(reasonCode: string, note: string | null) {
    if (submitting) return;
    const ok = await denyWithCode(dispute.id, reasonCode, note);
    if (!ok) return;
    setDenyOpen(false);
    showToast({ kind: "info", message: `Denied · ${dispute.id}` });
  }

  async function handleEscalate(note: string | null) {
    if (submitting) return;
    const ok = await escalate(dispute.id, note);
    if (!ok) return;
    setEscalateOpen(false);
    showToast({ kind: "info", message: `Escalated · ${dispute.id}` });
  }

  return (
    <aside
      aria-label="Admin actions"
      className={`${className} rounded-2xl border border-line bg-bone p-5 md:p-6`}
    >
      <Kicker as="p" tone="soft">
        Admin actions
      </Kicker>
      {submitting && <p role="status">{pendingAction(dispute.id)}</p>}
      <div className="mt-5 space-y-3">
        {isOpen && (
          <>
            <button
              type="button"
              onClick={() => setResolveOpen(true)}
              disabled={submitting}
              className={primaryFreshBtn}
            >
              Approve refund…
            </button>
            <button
              type="button"
              onClick={() => {
                setDenyCode(null);
                setDenyOpen(true);
              }}
              disabled={submitting}
              className={dangerBtn}
            >
              Deny…
            </button>
          </>
        )}
        {isOpen && dispute.refundable === false && dispute.refundBlockedMessage && (
          <p className="font-sans text-sm text-slate-soft">{dispute.refundBlockedMessage}</p>
        )}
        {isOpen && dispute.refundable === true && (
          <p className="font-sans text-sm text-slate-soft">A full refund can be approved now.</p>
        )}
        {isOpen && !isEscalated && (
          <button
            type="button"
            onClick={() => setEscalateOpen(true)}
            disabled={submitting}
            className={tertiaryBtn}
          >
            Escalate…
          </button>
        )}
        {!isOpen && dispute.refundStatus === "manual_review" && <AdminRefundPanel key={dispute.id} dispute={dispute} onConfirmOpenChange={setPanelDialogOpen} />}
        {!isOpen && dispute.refundStatus !== "manual_review" && (
          <p className="font-sans text-sm text-slate-soft">
            This dispute is closed. No further actions are available.
          </p>
        )}
      </div>

      <AdminResolveDisputeModal
        submitting={submitting}
        open={resolveOpen}
        escDisabled={pendingResolve !== null}
        onClose={() => {
          setResolveOpen(false);
          setPendingResolve(null);
        }}
        onContinue={handleContinueResolve}
        disputeId={dispute.id}
        orderTotal={dispute.orderSnapshot.total}
        itemChargeAmount={dispute.itemChargeAmount}
        paymentMethod={dispute.orderSnapshot.paymentMethod}
        refundable={dispute.refundable}
        refundBlockedReason={dispute.refundBlockedReason}
        refundBlockedMessage={dispute.refundBlockedMessage}
        paymentPhotoCount={dispute.paymentPhotoCount}
        reasonLabel={DISPUTE_REASON_LABEL[dispute.reason]}
        onDenyWithReason={handleSwapToDeny}
      />
      {pendingResolve &&
        pendingResolve.resolution !== "deny" &&
        pendingResolve.refundAmount !== null && (
          <AdminRefundConfirmModal
            submitting={submitting}
            open={true}
            onCancel={handleCancelRefund}
            onConfirm={handleConfirmRefund}
            amount={pendingResolve.refundAmount}
            runnerHandle={runnerHandle}
            isFull={pendingResolve.resolution === "refund_full"}
            paymentMethod={dispute.orderSnapshot.paymentMethod}
          />
        )}
      <AdminDenyDisputeModal
        submitting={submitting}
        open={denyOpen}
        onClose={() => setDenyOpen(false)}
        onSubmit={handleDeny}
        disputeId={dispute.id}
        defaultCode={denyCode}
      />
      <AdminEscalateModal
        submitting={submitting}
        open={escalateOpen}
        onClose={() => setEscalateOpen(false)}
        onSubmit={handleEscalate}
        targetLabel={`Escalate ${dispute.id}`}
        body="Push this dispute to the next review tier when you can't decide yet. You can still approve or deny it afterwards."
      />
    </aside>
  );
}

const primaryFreshBtn = `${BTN_PRIMARY} ${BTN_SIZE.sm} w-full`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} w-full`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm} w-full`;
