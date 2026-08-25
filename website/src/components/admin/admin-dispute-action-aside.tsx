"use client";

import { useState } from "react";
import { useAdminDisputeStore } from "@/store/admin-dispute-store";
import { useToast } from "@/hooks/use-toast";
import type { Dispute, DisputeResolution } from "@/lib/admin-disputes";
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
  const resolve = useAdminDisputeStore((s) => s.resolve);
  const deny = useAdminDisputeStore((s) => s.deny);
  const escalate = useAdminDisputeStore((s) => s.escalate);

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

  function handleContinueResolve(args: PendingResolve) {
    if (args.resolution === "deny") {
      // Deny path inside Resolve modal — no stacked confirm.
      resolve(dispute.id, {
        resolution: "deny",
        refundAmount: null,
        reason: args.note,
      });
      setResolveOpen(false);
      showToast({ kind: "info", message: `Closed without refund · ${dispute.id}` });
      return;
    }
    setPendingResolve(args);
  }

  function handleConfirmRefund() {
    if (!pendingResolve) return;
    resolve(dispute.id, {
      resolution: pendingResolve.resolution,
      refundAmount: pendingResolve.refundAmount,
      reason: pendingResolve.note,
    });
    setPendingResolve(null);
    setResolveOpen(false);
    showToast({
      kind: "success",
      message: `Refunded · ${dispute.id}`,
    });
  }

  function handleCancelRefund() {
    // Back to parent — pendingResolve cleared, parent state still set.
    setPendingResolve(null);
  }

  function handleDeny(reason: string) {
    deny(dispute.id, reason);
    setDenyOpen(false);
    showToast({ kind: "info", message: `Denied · ${dispute.id}` });
  }

  function handleEscalate(note: string | null) {
    escalate(dispute.id, note);
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
      <div className="mt-5 space-y-3">
        {isOpen && (
          <>
            <button
              type="button"
              onClick={() => setResolveOpen(true)}
              className={primaryFreshBtn}
            >
              Resolve…
            </button>
            <button
              type="button"
              onClick={() => setDenyOpen(true)}
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
            className={isOpen ? tertiaryBtn : secondaryBtn}
          >
            Escalate…
          </button>
        )}
        {!isOpen && !isDenied && (
          <p className="font-sans text-sm text-slate-soft">
            This dispute is closed. No further actions are available.
          </p>
        )}
      </div>

      <AdminResolveDisputeModal
        open={resolveOpen}
        escDisabled={pendingResolve !== null}
        onClose={() => {
          setResolveOpen(false);
          setPendingResolve(null);
        }}
        onContinue={handleContinueResolve}
        disputeId={dispute.id}
        orderTotal={dispute.orderSnapshot.total}
      />
      {pendingResolve &&
        pendingResolve.resolution !== "deny" &&
        pendingResolve.refundAmount !== null && (
          <AdminRefundConfirmModal
            open={true}
            onCancel={handleCancelRefund}
            onConfirm={handleConfirmRefund}
            amount={pendingResolve.refundAmount}
            runnerHandle={runnerHandle}
            isFull={pendingResolve.resolution === "refund_full"}
          />
        )}
      <AdminDenyDisputeModal
        open={denyOpen}
        onClose={() => setDenyOpen(false)}
        onSubmit={handleDeny}
        disputeId={dispute.id}
      />
      <AdminEscalateModal
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
