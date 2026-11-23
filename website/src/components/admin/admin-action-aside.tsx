"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { adminUserKeys, useAdminUserActions } from "@/hooks/use-admin-users";
import { useToast } from "@/hooks/use-toast";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { CONSENT_WITHDRAWAL_TEXT } from "@/lib/admin-relevance";
import { ApiError } from "@/lib/api";
import { clearRelevanceConsentWithdrawal, recordRelevanceConsentWithdrawal, type AdminUserDetail } from "@/lib/api-admin";
import { formatLongDate } from "@/lib/format";
import { useAuthStore } from "@/store/auth-store";
import { AdminConfirmModal } from "./admin-form-modal";
import { AdminRejectModal } from "./admin-reject-modal";
import { AdminResetVerificationModal } from "./admin-reset-verification-modal";
import { AdminSuspendModal } from "./admin-suspend-modal";
import { AdminMessageModal } from "./admin-message-modal";
import { Kicker } from "@/components/ui/kicker";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";

interface AdminActionAsideProps {
  row: AdminUserRow;
}

// Sticky-on-lg / inline-on-md action cluster for the photographer detail
// page. Carries Reject, Suspend, Unsuspend, Reset verification, Send
// message; profiles approve themselves once identity passes (Q7), so Approve
// shows only for a profile held after an admin reset or rejection. Modal
// state is local so the detail page stays a thin shell.
//
// Destructive actions (Reject, Suspend) carry the shared BTN_DANGER recipe
// (error text on bone-deep) so they read as consequential before the modal
// opens; restorative/neutral actions stay secondary/ghost.
export function AdminActionAside({ row }: AdminActionAsideProps) {
  const { showToast } = useToast();
  const { approveHeld, reject, suspend, unsuspend, resetVerification, message, messageRecoveryNotice, isPending: actionPending, pendingAction } = useAdminUserActions();
  const submitting = actionPending([row.userId]);
  const activeLabel = pendingAction(row.userId);
  // The withdrawal call (RV-L7) locks the other actions too, as theirs lock it.
  const [withdrawalBusy, setWithdrawalBusy] = useState(false);
  const locked = submitting || withdrawalBusy;

  const [rejectOpen, setRejectOpen] = useState(false);
  const [suspendOpen, setSuspendOpen] = useState(false);
  const [messageOpen, setMessageOpen] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);
  const [reviewRevision, setReviewRevision] = useState(row.verificationRevision);

  const isPending = row.verificationStatus === "pending";
  const isSuspended = row.suspendedAt !== null;
  const isIncomplete = row.verificationStatus === "incomplete";
  const displayName = row.brandName ?? row.name;

  // PD5: the same approve the verifications drawer offers a profile held after a reset or rejection.
  async function handleApproveHeld() {
    if (await approveHeld(row.userId, row.verificationRevision)) {
      showToast({ kind: "success", message: `Hold released · ${displayName}` });
    }
  }

  async function handleReject(reason: string) {
    const ok = await reject(row.userId, reason, reviewRevision);
    if (ok) {
      setRejectOpen(false);
      showToast({ kind: "success", message: `Sent back · ${displayName}` });
    }
    return ok;
  }

  async function handleSuspend(reason: string) {
    const ok = await suspend(row.userId, reason);
    if (ok) {
      setSuspendOpen(false);
      showToast({ kind: "success", message: `${displayName} is suspended. They were not emailed.` });
    }
    return ok;
  }

  async function handleUnsuspend() {
    const ok = await unsuspend(row.userId);
    if (ok) {
      showToast({ kind: "success", message: `${displayName} can sign in again. They are told in their inbox.` });
    }
  }

  async function handleReset(reason: string) {
    const ok = await resetVerification(row.userId, reason, reviewRevision);
    if (ok) {
      setResetOpen(false);
      showToast({
        kind: "success",
        message: `Verification reset · ${displayName}`,
      });
    }
    return ok;
  }

  async function handleMessage(payload: { subject: string; body: string }) {
    const ok = await message(row.userId, payload);
    if (ok) {
      setMessageOpen(false);
      showToast({
        kind: "success",
        message: `Message sent · ${displayName}`,
      });
    }
    return ok;
  }

  return (
    <aside
      aria-label="Admin actions"
      aria-busy={submitting}
      className="lg:sticky lg:top-24 lg:col-start-2 lg:row-start-1 mb-10 lg:mb-0 rounded-2xl border border-line bg-bone p-5 md:p-6"
    >
      <Kicker as="p" tone="soft">
        Admin actions
      </Kicker>
      <div className="mt-5 space-y-3">
        {isPending && row.adminHold && (
          <button
            type="button"
            disabled={locked}
            onClick={handleApproveHeld}
            className={primaryBtn}
          >
            {activeLabel === "Approving…" ? activeLabel : "Approve"}
          </button>
        )}

        {isPending && (
          <button
            type="button"
            disabled={locked}
            onClick={() => { setReviewRevision(row.verificationRevision); setRejectOpen(true); }}
            className={dangerBtn}
          >
            Send back…
          </button>
        )}

        {!isSuspended ? (
          <button
            type="button"
            disabled={locked}
            onClick={() => setSuspendOpen(true)}
            className={dangerBtn}
          >
            Suspend…
          </button>
        ) : (
          <button
            type="button"
            disabled={locked}
            onClick={handleUnsuspend}
            className={secondaryBtn}
          >
            {activeLabel === "Unsuspending…" ? activeLabel : "Unsuspend"}
          </button>
        )}

        {!isIncomplete && (
          <button
            type="button"
            disabled={locked}
            onClick={() => { setReviewRevision(row.verificationRevision); setResetOpen(true); }}
            className={secondaryBtn}
          >
            Reset verification…
          </button>
        )}

        <button
          type="button"
          disabled={locked}
          onClick={() => setMessageOpen(true)}
          className={tertiaryBtn}
        >
          Send message…
        </button>
      </div>

      {isSuspended && row.suspensionReason && (
        <div className="mt-6 pt-5 border-t border-line">
          <Kicker as="p" tone="soft">
            Suspension reason
          </Kicker>
          <p className="font-sans text-sm text-ink-soft mt-2">
            {row.suspensionReason}
          </p>
        </div>
      )}

      {row.role === "PHOTOGRAPHER" && <ConsentWithdrawal row={row} locked={locked} busy={withdrawalBusy} setBusy={setWithdrawalBusy} />}

      <AdminRejectModal

        submitting={submitting}
        open={rejectOpen}
        onClose={() => setRejectOpen(false)}
        onSubmit={handleReject}
        photographerName={displayName}
      />
      <AdminResetVerificationModal
        submitting={submitting}
        open={resetOpen}
        onClose={() => setResetOpen(false)}
        onSubmit={handleReset}
        photographerName={displayName}
      />
      <AdminSuspendModal
        submitting={submitting}
        open={suspendOpen}
        onClose={() => setSuspendOpen(false)}
        onSubmit={handleSuspend}
        photographerName={displayName}
      />
      <AdminMessageModal
        recoveryNotice={messageRecoveryNotice(row.userId)}
        submitting={submitting}
        open={messageOpen}
        onClose={() => setMessageOpen(false)}
        onSubmit={handleMessage}
        photographerName={displayName}
      />
    </aside>
  );
}

// RV-L7 part c: record or clear the photographer's e-mailed withdrawal of the consent to automatic hides and
// holds. The state shown is always the server's: the answer is written into the detail cache, then refetched.
function ConsentWithdrawal({ row, locked, busy, setBusy }: { row: AdminUserRow; locked: boolean; busy: boolean; setBusy: (busy: boolean) => void }) {
  const T = CONSENT_WITHDRAWAL_TEXT;
  const qc = useQueryClient();
  const { showToast } = useToast();
  const headingId = useId();
  const [open, setOpen] = useState(false);
  const actionRef = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  const withdrawnAt = row.relevanceConsentWithdrawnAt ?? null;
  const withdrawn = withdrawnAt !== null;

  // The shared Modal does not return focus; the one action button keeps its element across both states.
  useEffect(() => {
    if (wasOpen.current && !open) actionRef.current?.focus();
    wasOpen.current = open;
  }, [open]);

  async function handleConfirm() {
    if (busy) return;
    setBusy(true);
    try {
      const answer = await (withdrawn ? clearRelevanceConsentWithdrawal : recordRelevanceConsentWithdrawal)(row.userId);
      const key = adminUserKeys.detail(useAuthStore.getState().user?.id ?? null, row.userId);
      qc.setQueryData<AdminUserDetail | null>(key, (detail) =>
        detail ? { ...detail, relevanceConsentWithdrawnAt: answer.consentWithdrawnAt } : detail);
      void qc.invalidateQueries({ queryKey: key }); // the decision log gained an entry
      setOpen(false);
      showToast({ kind: "success", message: withdrawn ? T.cleared : T.recorded });
    } catch (error) {
      showToast({ kind: "error", message: error instanceof ApiError
        ? error.errors[0]?.message ?? error.message
        : withdrawn ? T.clearFailed : T.recordFailed });
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby={headingId} className="mt-6 pt-5 border-t border-line">
      <Kicker as="p" tone="soft" id={headingId}>
        {T.title}
      </Kicker>
      {!withdrawn && <p className="font-sans text-sm text-slate mt-2">{T.caption}</p>}
      <p className="font-sans text-sm text-ink-soft mt-2">
        {withdrawn
          ? <>{T.withdrawnOn} <span className="tnum whitespace-nowrap">{formatLongDate(withdrawnAt)}</span>. {T.withdrawnAfter}</>
          : T.notWithdrawn}
      </p>
      <button ref={actionRef} type="button" disabled={locked} onClick={() => setOpen(true)} className={`${secondaryBtn} mt-4`}>
        {withdrawn ? T.clear : T.record}
      </button>
      <AdminConfirmModal
        open={open}
        submitting={busy}
        pendingLabel={withdrawn ? T.clearing : T.recording}
        onClose={() => setOpen(false)}
        onConfirm={handleConfirm}
        title={withdrawn ? T.clearTitle : T.recordTitle}
        confirmLabel={withdrawn ? T.clearConfirmButton : T.recordConfirmButton}
      >
        <p className="font-sans text-sm text-ink-soft">{withdrawn ? T.clearConfirm : T.recordConfirm}</p>
      </AdminConfirmModal>
    </section>
  );
}

const primaryBtn = `${BTN_PRIMARY} ${BTN_SIZE.sm} w-full`;
const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm} w-full`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} w-full`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm} w-full`;
