"use client";

import { useEffect, useState } from "react";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { AdminConfirmModal, AdminFormModal } from "@/components/admin/admin-form-modal";
import { BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useDisputeActions } from "@/hooks/use-admin-disputes";
import { useToast } from "@/hooks/use-toast";
import type { Dispute } from "@/lib/admin-disputes";
import { formatLongDate, formatSalesDate } from "@/lib/format";
import { isRefundLinkExpired } from "@/lib/refund-helpers";
import { copyToClipboard, formatPrice } from "@/lib/utils";

const sm = BTN_SIZE.sm;

// PayMongo cannot refund QR Ph through its API, so a parked refund
// (refundStatus "manual_review") is finished by hand: the admin makes a
// refund link in the PayMongo dashboard, sends it to the runner here, then
// marks the refund done. The server checks the link's prefix against config,
// so the browser only requires a non-empty value.
export function AdminRefundPanel({
  dispute,
  onConfirmOpenChange,
}: {
  dispute: Dispute;
  // Lets a host drawer stand its own Esc handler down while the confirm is up.
  onConfirmOpenChange?: (open: boolean) => void;
}) {
  const { showToast } = useToast();
  const { sendRefundLink, markRefunded, deny, isPending } = useDisputeActions();
  const busy = isPending(dispute.id);
  const [url, setUrl] = useState("");
  const [replacing, setReplacing] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [cancelOpen, setCancelOpen] = useState(false);
  const [cancelReason, setCancelReason] = useState("");
  // The cleanup also clears the host flag if the panel unmounts with a dialog open.
  useEffect(() => {
    onConfirmOpenChange?.(confirmOpen || cancelOpen);
    return () => onConfirmOpenChange?.(false);
  }, [confirmOpen, cancelOpen, onConfirmOpenChange]);

  const savedLink = dispute.refundLinkUrl;
  const showField = !savedLink || replacing;
  const trimmed = url.trim();
  // The server never auto-cancels once a link was sent (the runner may have claimed it): past this date an admin checks PayMongo.
  const linkSent = !!dispute.refundLinkSentAt;
  const checkPaymongo = linkSent && isRefundLinkExpired(dispute.refundCancelsAt);

  async function handleCopy(text: string) {
    const ok = await copyToClipboard(text);
    showToast(ok
      ? { kind: "success", message: "Payment ID copied" }
      : { kind: "error", message: "Could not copy. Select the payment ID and copy it by hand." });
  }

  async function handleSend() {
    if (busy || !trimmed) return;
    if (trimmed === savedLink) {
      showToast({ kind: "info", message: "This link was already sent." });
      return;
    }
    if (!(await sendRefundLink(dispute.id, trimmed))) return;
    setUrl("");
    setReplacing(false);
    showToast({ kind: "success", message: `Refund link sent · ${dispute.id}` });
  }

  async function handleMarkRefunded() {
    if (busy) return;
    if (!(await markRefunded(dispute.id))) return;
    setConfirmOpen(false);
    showToast({ kind: "success", message: `Marked refunded · ${dispute.id}` });
  }

  // Deny is the cancel route: the server allows it here only past the cancel date and needs a reason.
  async function handleCancelRefund() {
    const reason = cancelReason.trim();
    if (busy || !reason) return;
    if (!(await deny(dispute.id, reason))) return;
    setCancelOpen(false);
    setCancelReason("");
    showToast({ kind: "success", message: `Refund cancelled · ${dispute.id}` });
  }

  return (
    <div className="space-y-4 font-sans text-sm text-ink-soft">
      {dispute.paymentId ? (
        <div className="space-y-1">
          <p className="text-ink">Payment ID</p>
          <div className="flex items-center justify-between gap-3">
            <span className="font-mono tnum break-all text-ink">{dispute.paymentId}</span>
            <button type="button" aria-label="Copy payment ID" onClick={() => handleCopy(dispute.paymentId!)} disabled={busy} className={`${BTN_GHOST} ${sm}`}>
              Copy
            </button>
          </div>
        </div>
      ) : (
        <p>
          Search these in PayMongo.{" "}
          <span className="font-mono tnum break-all text-ink">{dispute.paymentRef ?? "—"}</span>
          {" · "}
          <span className="tnum text-ink">{formatPrice(dispute.orderSnapshot.total)}</span>
          {" · "}
          <span data-testid="refund-paid-at" className="tnum">{formatSalesDate(dispute.orderSnapshot.paidAt)}</span>
        </p>
      )}

      {dispute.refundWindowClosesAt && dispute.refundCancelsAt && (
        linkSent ? (
          <p>
            PayMongo window closes {formatLongDate(dispute.refundWindowClosesAt)}. A refund with a sent link is never
            cancelled automatically: after {formatLongDate(dispute.refundCancelsAt)}, check PayMongo.
          </p>
        ) : (
          <p>
            PayMongo window closes {formatLongDate(dispute.refundWindowClosesAt)}. QuickPitik cancels this
            refund on {formatLongDate(dispute.refundCancelsAt)} if it is not done.
          </p>
        )
      )}

      {checkPaymongo && (
        <div className="space-y-1 rounded-2xl border border-line-strong bg-bone-deep px-4 py-3">
          <p className="font-semibold text-ink">Check PayMongo</p>
          <p>If the refund was claimed, mark it refunded. If the link is still unclaimed, cancel the refund.</p>
        </div>
      )}

      {showField ? (
        <div className="space-y-2">
          <label htmlFor={`refund-link-${dispute.id}`} className="block text-ink">Refund link</label>
          <input
            id={`refund-link-${dispute.id}`}
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={busy}
            className="w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button type="button" onClick={handleSend} disabled={busy || !trimmed} className={`${BTN_SECONDARY} ${sm} w-full`}>
            Send to runner
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {isRefundLinkExpired(dispute.refundLinkExpiresAt) ? (
            <p className="text-ink">
              Refund link expired · <span className="tnum">{formatSalesDate(dispute.refundLinkExpiresAt!)}</span>
            </p>
          ) : (
            <p className="text-ink">
              Refund link sent
              {dispute.refundLinkSentAt && <> · <span className="tnum">{formatSalesDate(dispute.refundLinkSentAt)}</span></>}
            </p>
          )}
          <p className="font-mono break-all text-ink">{savedLink}</p>
          <button type="button" onClick={() => setReplacing(true)} disabled={busy} className={`${BTN_GHOST} ${sm}`}>
            Send a new link
          </button>
        </div>
      )}

      <button type="button" onClick={() => setConfirmOpen(true)} disabled={busy} className={`${BTN_SECONDARY} ${sm} w-full`}>
        Mark refunded
      </button>

      {checkPaymongo && (
        <button type="button" onClick={() => setCancelOpen(true)} disabled={busy} className={`${BTN_DANGER} ${sm} w-full`}>
          Cancel refund…
        </button>
      )}

      <AdminFormModal
        open={cancelOpen}
        submitting={busy}
        pendingLabel="Cancelling…"
        onClose={() => setCancelOpen(false)}
        onSubmit={handleCancelRefund}
        title="Cancel refund"
        intro="Only do this after PayMongo shows the refund link unclaimed. The runner sees your reason."
        submitLabel="Cancel refund"
        submitDisabled={!cancelReason.trim()}
        cancelLabel="Keep refund"
      >
        <AdminTextarea
          id={`cancel-refund-reason-${dispute.id}`}
          label="Reason (required, max 500)"
          value={cancelReason}
          onChange={setCancelReason}
          maxLength={500}
        />
      </AdminFormModal>

      <AdminConfirmModal
        open={confirmOpen}
        submitting={busy}
        pendingLabel="Marking refunded…"
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleMarkRefunded}
        title="Mark refunded"
        confirmLabel="Mark refunded"
      >
        <p className="font-sans text-sm text-ink-soft">
          Only do this after PayMongo shows this payment as Refunded.
        </p>
      </AdminConfirmModal>
    </div>
  );
}
