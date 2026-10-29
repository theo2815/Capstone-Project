"use client";

import { useState } from "react";
import { AdminConfirmModal } from "@/components/admin/admin-form-modal";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { useDisputeActions } from "@/hooks/use-admin-disputes";
import { useToast } from "@/hooks/use-toast";
import type { Dispute } from "@/lib/admin-disputes";
import { formatLongDate, formatSalesDate } from "@/lib/format";
import { copyToClipboard, formatPrice } from "@/lib/utils";

const sm = BTN_SIZE.sm;

// PayMongo cannot refund QR Ph through its API, so a parked refund
// (refundStatus "manual_review") is finished by hand: the admin makes a
// refund link in the PayMongo dashboard, sends it to the runner here, then
// marks the refund done. The server checks the link's prefix against config,
// so the browser only requires a non-empty value.
export function AdminRefundPanel({ dispute }: { dispute: Dispute }) {
  const { showToast } = useToast();
  const { sendRefundLink, markRefunded, isPending } = useDisputeActions();
  const busy = isPending(dispute.id);
  const [url, setUrl] = useState("");
  const [replacing, setReplacing] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  const savedLink = dispute.refundLinkUrl;
  const showField = !savedLink || replacing;
  const trimmed = url.trim();

  async function handleCopy(text: string) {
    const ok = await copyToClipboard(text);
    showToast(ok
      ? { kind: "success", message: "Payment ID copied" }
      : { kind: "error", message: "Could not copy. Select the payment ID and copy it by hand." });
  }

  async function handleSend() {
    if (busy || !trimmed) return;
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

  return (
    <div className="space-y-4 font-sans text-sm text-ink-soft">
      {dispute.paymentId ? (
        <div className="flex items-center justify-between gap-3">
          <span className="font-mono tnum break-all text-ink">{dispute.paymentId}</span>
          <button type="button" onClick={() => handleCopy(dispute.paymentId!)} disabled={busy} className={`${BTN_GHOST} ${sm}`}>
            Copy
          </button>
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
        <p>
          PayMongo window closes {formatLongDate(dispute.refundWindowClosesAt)}. QuickPitik cancels this
          refund on {formatLongDate(dispute.refundCancelsAt)} if it is not done.
        </p>
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
            className="w-full border-b border-line bg-transparent py-2 text-ink focus:border-fresh disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button type="button" onClick={handleSend} disabled={busy || !trimmed} className={`${BTN_SECONDARY} ${sm} w-full`}>
            Send to runner
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          <p className="text-ink">
            Refund link sent
            {dispute.refundLinkSentAt && <> · <span className="tnum">{formatSalesDate(dispute.refundLinkSentAt)}</span></>}
          </p>
          <p className="font-mono break-all text-ink">{savedLink}</p>
          <button type="button" onClick={() => setReplacing(true)} disabled={busy} className={`${BTN_GHOST} ${sm}`}>
            Send a new link
          </button>
        </div>
      )}

      <button type="button" onClick={() => setConfirmOpen(true)} disabled={busy} className={`${BTN_SECONDARY} ${sm} w-full`}>
        Mark refunded
      </button>

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
