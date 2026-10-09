"use client";

import { useState } from "react";
import { useDisputeActions } from "@/hooks/use-admin-disputes";
import { useToast } from "@/hooks/use-toast";
import type { Dispute, DisputeResolution } from "@/lib/admin-disputes";
import { formatPrice } from "@/lib/utils";
import { AdminResolveDisputeModal } from "./admin-resolve-dispute-modal";
import { AdminRefundConfirmModal } from "./admin-refund-confirm-modal";
import { AdminDenyDisputeModal } from "./admin-deny-dispute-modal";
import { AdminEscalateModal } from "./admin-escalate-modal";
import { Kicker } from "@/components/ui/kicker";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";

interface AdminDisputeActionAsideProps {
  dispute: Dispute;
  runnerHandle: string;
}

interface PendingResolve {
  resolution: DisputeResolution;
  refundAmount: number | null;
  note: string;
}

// Sticky-on-lg / inline-on-md action cluster for the dispute detail
// page. Carries Resolve / Deny / Escalate. Resolve uses a stacked-modal
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
}: AdminDisputeActionAsideProps) {
  const { showToast } = useToast();
  const { resolve, deny, escalate, isPending, pendingAction } = useDisputeActions();

  const submitting = isPending(dispute.id);
  const [resolveOpen, setResolveOpen] = useState(false);
  const [denyOpen, setDenyOpen] = useState(false);
  const [escalateOpen, setEscalateOpen] = useState(false);

  // Lifted state for the stacked refund-confirm flow.
  const [pendingResolve, setPendingResolve] = useState<PendingResolve | null>(
    null,
  );

  const isOpen = dispute.status === "open";
  const isDenied = dispute.status === "denied";
  const escalateAllowed = isOpen || isDenied;

  async function handleContinueResolve(args: PendingResolve) {
    if (submitting) return;
    if (args.resolution === "deny") {
      // Deny path inside Resolve modal — no stacked confirm.
      const ok = await resolve(dispute.id, {
        resolution: "deny",
        refundAmount: null,
        reason: args.note,
      });
      if (!ok) return;
      setResolveOpen(false);
      showToast({ kind: "info", message: `Closed without refund · ${dispute.id}` });
      return;
    }
    setPendingResolve(args);
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

  async function handleDeny(reason: string) {
    if (submitting) return;
    const ok = await deny(dispute.id, reason);
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
      className="lg:sticky lg:top-24 mt-12 lg:mt-0 rounded-2xl border border-line bg-bone p-5 md:p-6"
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
              Resolve…
            </button>
            <button
              type="button"
              onClick={() => setDenyOpen(true)}
              disabled={submitting}
              className={dangerBtn}
            >
              Deny…
            </button>
          </>
        )}
        {escalateAllowed && (
          <button
            type="button"
            onClick={() => setEscalateOpen(true)}
            disabled={submitting}
            className={isOpen ? tertiaryBtn : secondaryBtn}
          >
            Escalate…
          </button>
        )}
        {!isOpen && !isDenied && (
          <p className="font-sans text-sm text-slate-soft">
            {dispute.refundStatus === "manual_review"
              ? "Review the payment in PayMongo before another refund attempt."
              : "This dispute is closed. No further actions are available."}
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
          />
        )}
      <AdminDenyDisputeModal
        submitting={submitting}
        open={denyOpen}
        onClose={() => setDenyOpen(false)}
        onSubmit={handleDeny}
        disputeId={dispute.id}
      />
      <AdminEscalateModal
        submitting={submitting}
        open={escalateOpen}
        onClose={() => setEscalateOpen(false)}
        onSubmit={handleEscalate}
        targetLabel={`Escalate ${dispute.id}`}
        body="Push this dispute to the next review tier. The runner sees the dispute remains open while higher-level review takes place."
      />
    </aside>
  );
}

const primaryFreshBtn = `${BTN_PRIMARY} ${BTN_SIZE.sm} w-full`;
const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm} w-full`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} w-full`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm} w-full`;
