"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { useToast } from "@/hooks/use-toast";
import { withdrawPhotographerPayout } from "@/lib/api-photographer-earnings";
import type { PhotographerPayout } from "@/lib/photographer-mock";

// Confirmation modal for cancelling a held payout request. BE only allows this
// when the cycle is on hold — the hero gates the UI to that state. On confirm,
// the cycle is soft-cancelled (status → cancelled; the row and its audit history
// persist) and the balance cache is invalidated so the photographer can submit
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
        message: "Request cancelled. You can submit a new one when ready.",
      });
      onClose();
    } catch (err) {
      console.error("[payouts] cancel failed", err);
      showToast({
        kind: "error",
        message: "Could not cancel request. Please try again.",
      });
    } finally {
      setSubmitting(false);
    }
  }

  if (!openRequest) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Cancel payout request">
      <div className="space-y-6">
        <div>
          <Kicker as="p" tone="soft">
            Request to cancel
          </Kicker>
          <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink tnum mt-2">
            ₱{openRequest.amount.toLocaleString()}
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate mt-2">
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
          Cancelling removes this request. Your unpaid balance stays on your
          account — submit a new request once you&apos;ve addressed the reason
          above.
        </p>

        <div className="flex flex-wrap items-center justify-end gap-3 pt-2 border-t border-line">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2"
          >
            Keep request
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-error border border-error hover:bg-error hover:text-surface disabled:opacity-50 disabled:cursor-not-allowed transition-colors rounded-full px-5 py-2"
          >
            {submitting ? "Cancelling…" : "Cancel request"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
