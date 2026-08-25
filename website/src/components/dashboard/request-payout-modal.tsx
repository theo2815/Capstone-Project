"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import {
  requestPhotographerPayout,
  type PayoutBalanceResponse,
} from "@/lib/api-photographer-earnings";
import { formatPayoutNumber } from "@/lib/payout-format";
import { PAYOUT_METHOD_LABEL } from "@/store/photographer-settings-store";

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
  // Primary account comes from the BE balance response (authoritative), not
  // the client settings store — a stale store primary the DB lacks is what
  // produced the demo's PAYOUT_NO_ACCOUNT on submit.
  const primary = balance?.primaryAccount ?? null;
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
      // Surface the backend's specific reason (PAYOUT_NO_ACCOUNT /
      // PAYOUT_BELOW_MINIMUM / PAYOUT_REQUEST_OPEN — already user-friendly)
      // instead of a generic line that hides why the request failed.
      showToast({
        kind: "error",
        message:
          err instanceof ApiError
            ? (err.errors[0]?.message ??
              "Could not submit request. Please try again.")
            : "Could not submit request. Please try again.",
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
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink disabled:opacity-50 disabled:cursor-not-allowed transition-colors px-4 py-2"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting || !primary}
            className={`${BTN_PRIMARY} ${BTN_SIZE.sm}`}
          >
            {submitting ? "Submitting…" : "Submit request"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
