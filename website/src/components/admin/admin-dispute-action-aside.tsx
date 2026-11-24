"use client";

import { useLayoutEffect, useState, type ReactNode } from "react";
import { useDisputeActions } from "@/hooks/use-admin-disputes";
import { useToast } from "@/hooks/use-toast";
import { DISPUTE_REASON_LABEL, type Dispute, type DisputeResolution } from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";
import { AdminResolveDisputeModal } from "./admin-resolve-dispute-modal";
import { AdminRefundConfirmModal } from "./admin-refund-confirm-modal";
import { AdminDenyDisputeModal } from "./admin-deny-dispute-modal";
import { AdminEscalateModal } from "./admin-escalate-modal";
import { AdminRefundPanel, RefundPaymentReference } from "./admin-refund-panel";
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
  // Phones: the drawer body goes here, and the buttons move to a row stuck to the bottom of the drawer's scroll after it.
  body?: ReactNode;
  // Called once the dispute has left the open queue (denied, or refunded in full), so a host drawer can close.
  onDecided?: () => void;
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
  body,
  onDecided,
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
  // A failed refund can be approved again or denied, even on a dispute that was escalated first (F4).
  const refundFailed = dispute.attention === "refund_problem" && dispute.refundStatus === "failed";
  // Only a parked refund shows the refund panel; the other refund problems show the payment and the next step.
  const refundProblem = dispute.attention === "refund_problem" && dispute.refundStatus !== "manual_review";
  // A stuck refund is still in flight: the server refuses every decision until PayMongo answers (F4).
  const refundStuck = refundProblem && !refundFailed;
  const isOpen = (dispute.status === "open" || isEscalated || refundFailed) && !refundStuck;

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
      kind: "success",
      message: refunded
        ? `Refunded ${formatPrice(confirmed.refundAmount!)} · ${confirmed.id}`
        : dispute.orderSnapshot.paymentMethod === "qrph"
          ? "Refund approved. Next, send the runner the refund link."
          : "Refund approved. PayMongo is sending it, and the dispute closes when it is confirmed.",
    });
    // Not yet resolved (a QR Ph link still to send, or PayMongo still sending): the drawer stays on the next step.
    if (confirmed.status === "resolved") onDecided?.();
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
    showToast({ kind: "success", message: "Denied. The runner was sent your reason. It is now under Closed." });
    onDecided?.();
  }

  async function handleEscalate(note: string | null) {
    if (submitting) return;
    const ok = await escalate(dispute.id, note);
    if (!ok) return;
    setEscalateOpen(false);
    showToast({ kind: "success", message: "Escalated. It stays under Needs you." });
  }

  const approveDeny = isOpen && (
    <>
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
      <button
        type="button"
        onClick={() => setResolveOpen(true)}
        disabled={submitting}
        className={primaryFreshBtn}
      >
        Approve refund…
      </button>
    </>
  );
  // The server escalates open disputes only.
  const escalateBtn = dispute.status === "open" && !refundStuck && (
    <button
      type="button"
      onClick={() => setEscalateOpen(true)}
      disabled={submitting}
      className={tertiaryBtn}
    >
      Escalate…
    </button>
  );

  return (
    <>
    <aside
      aria-label="Admin actions"
      className={`${className} rounded-2xl border border-line bg-bone p-5 md:p-6`}
    >
      <Kicker as="p" tone="soft">
        Admin actions
      </Kicker>
      {submitting && <p role="status">{pendingAction(dispute.id)}</p>}
      <div className="mt-5 space-y-3">
        {/* Same order as every footer: Escalate, Deny, then the green Approve refund last. */}
        {!body && escalateBtn}
        {!body && approveDeny}
        {isOpen && dispute.refundable === false && dispute.refundBlockedMessage && (
          <p className="font-sans text-sm text-slate-soft">{dispute.refundBlockedMessage}</p>
        )}
        {isOpen && dispute.refundable === true && (
          <p className="font-sans text-sm text-slate-soft">A full refund can be approved now.</p>
        )}
        {refundProblem && (
          <div className="space-y-3 font-sans text-sm text-ink-soft">
            <RefundPaymentReference dispute={dispute} disabled={submitting} />
            <p>
              {refundFailed
                ? "The refund failed. You can approve the refund again, or deny the dispute."
                : "PayMongo has not confirmed this refund yet. QuickPitik keeps checking by itself. Look up the payment in PayMongo if it stays like this."}
            </p>
          </div>
        )}
        {!isOpen && dispute.refundStatus === "manual_review" && <AdminRefundPanel key={dispute.id} dispute={dispute} onConfirmOpenChange={setPanelDialogOpen} />}
        {!isOpen && !refundProblem && dispute.refundStatus !== "manual_review" && (
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
        body="This marks the dispute as escalated and tells the runner and the photographer it needs more review. It stays in this queue with the same due date, and you can still approve or deny it afterwards."
      />
    </aside>
    {body}
    {/* -mx-6/px-6 cover the drawer's side padding so the scrolling evidence does not show beside the row.
        Tighter padding keeps the three labels on one line from 360 px; narrower screens wrap to a second line.
        The bottom padding clears a phone's home bar (the layout sets viewport-fit=cover). */}
    {body && isOpen && (
      <div
        role="group"
        aria-label="Dispute actions"
        className="sticky bottom-0 -mx-6 mt-10 flex flex-wrap gap-1.5 border-t border-line bg-bone px-6 pt-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))] [&>button]:w-auto [&>button]:flex-auto [&>button]:px-2"
      >
        {escalateBtn}
        {approveDeny}
      </div>
    )}
    </>
  );
}

const primaryFreshBtn = `${BTN_PRIMARY} ${BTN_SIZE.sm} w-full`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} w-full`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm} w-full`;
