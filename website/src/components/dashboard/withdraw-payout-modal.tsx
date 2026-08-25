"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { useToast } from "@/hooks/use-toast";
import { withdrawPhotographerPayout } from "@/lib/api-photographer-earnings";
import type { PhotographerPayout } from "@/lib/photographer-mock";

// Confirmation modal for withdrawing a held payout request. BE only allows
// withdraw when the cycle is on hold — this UI is gated to that state by the
// hero. On confirm, hard-deletes the cycle (audit log row persists) and
// invalidates the balance cache so the photographer can immediately submit
// a new request once they've addressed the hold reason.

export function WithdrawPayoutModal({
  isOpen,
  openRequest,
  onClose,
}: {
  isOpen: boolean;
  openRequest: PhotographerPayout | null;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    if (!openRequest || submitting) return;
    setSubmitting(true);
    try {
      await withdrawPhotographerPayout(openRequest.id);
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["photographer", "payouts", "balance"],
        }),
        queryClient.invalidateQueries({
          queryKey: ["photographer", "payouts"],
        }),
      ]);
      showToast({
        kind: "success",
        message: "Request withdrawn. You can submit a new one when ready.",
      });
      onClose();
    } catch (err) {
      console.error("[payouts] withdraw failed", err);
      showToast({
        kind: "error",
        message: "Could not withdraw request. Please try again.",
      });
    } finally {
      setSubmitting(false);
    }
  }

  if (!openRequest) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Withdraw payout request">
      <div className="space-y-6">
        <div>
          <Kicker as="p" tone="soft">
            Request to withdraw
          </Kicker>
          <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink tnum mt-2">
            ₱{openRequest.amount.toLocaleString()}
          </p>
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mt-2">
            {openRequest.id}
          </p>
        </div>

        {openRequest.holdReason && (
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <Kicker as="p" tone="soft">
              Admin&apos;s hold reason
            </Kicker>
            <p className="font-sans text-sm text-ink-soft mt-2">
              {openRequest.holdReason}
            </p>
          </div>
        )}

        <p className="font-sans text-sm text-slate">
          Withdrawing removes this request. The unpaid balance stays on your
          account — submit a new request once you&apos;ve addressed the reason
          above.
        </p>

        <div className="flex flex-wrap items-center justify-end gap-3 pt-2 border-t border-line">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-error border border-error hover:bg-error hover:text-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-full px-5 py-2"
          >
            {submitting ? "Withdrawing…" : "Withdraw request"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
