"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { useToast } from "@/hooks/use-toast";
import {
  requestPhotographerPayout,
  type PayoutBalanceResponse,
} from "@/lib/api-photographer-earnings";
import { formatPayoutNumber } from "@/lib/payout-format";
import { PAYOUT_METHOD_LABEL, usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// Confirmation modal before submitting a payout request. Shows: amount,
// primary payout account snapshot, and the one-open-at-a-time rule. POSTs
// to /me/photographer/payouts/request and invalidates the balance + payouts
// caches so the hero flips into "open request" state immediately. The BE
// enforces all the rules (₱500 min, primary account exists, no other open
// request) — this modal is just a friendly preview.

export function RequestPayoutModal({
  isOpen,
  balance,
  onClose,
}: {
  isOpen: boolean;
  balance: PayoutBalanceResponse | null;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const primary = usePhotographerSettingsStore((s) =>
    s.payouts.find((p) => p.isPrimary),
  );
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    if (submitting) return;
    setSubmitting(true);
    try {
      await requestPhotographerPayout();
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
        message: "Payout request submitted. Admin will review shortly.",
      });
      onClose();
    } catch (err) {
      console.error("[payouts] request failed", err);
      showToast({
        kind: "error",
        message: "Could not submit request. Please try again.",
      });
    } finally {
      setSubmitting(false);
    }
  }

  if (!balance) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Confirm payout request">
      <div className="space-y-6">
        <div>
          <Kicker as="p" tone="soft">
            Amount to request
          </Kicker>
          <p className="font-display text-4xl md:text-5xl font-semibold tracking-tight text-fresh tnum mt-2 leading-none">
            ₱{balance.unpaidBalance.toLocaleString()}
          </p>
        </div>

        {primary ? (
          <div className="rounded-xl border border-line bg-bone-deep p-4">
            <Kicker as="p" tone="soft">
              Sent to
            </Kicker>
            <p className="font-sans text-base text-ink mt-1">
              {primary.accountName}
            </p>
            <p className="font-mono text-sm text-ink tnum mt-1">
              {PAYOUT_METHOD_LABEL[primary.method]}
              <span className="text-slate-soft"> · </span>
              <span className="break-all">
                {formatPayoutNumber(primary.method, primary.accountNumber)}
              </span>
            </p>
          </div>
        ) : (
          <p className="font-sans text-sm text-error">
            No primary payout account configured. Set one up in Settings before
            requesting.
          </p>
        )}

        <p className="font-sans text-sm text-slate">
          Once submitted, admin reviews and transfers the funds manually. You
          can&apos;t submit another request until this one is paid or held. If
          held, you can withdraw and re-submit after fixing what admin
          flagged.
        </p>

        <div className="flex flex-wrap items-center justify-end gap-3 pt-2 border-t border-line">
          <button
            type="button"
            onClick={onClose}
            disabled={submitting}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink disabled:opacity-40 disabled:cursor-not-allowed transition-colors px-4 py-2"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting || !primary}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-bone bg-fresh hover:bg-fresh-deep disabled:opacity-40 disabled:cursor-not-allowed transition-colors rounded-full px-5 py-2"
          >
            {submitting ? "Submitting…" : "Submit request"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
