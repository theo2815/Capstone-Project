"use client";

import { useRef, useState } from "react";
import { useAdminUserActions } from "@/hooks/use-admin-users";
import { useToast } from "@/hooks/use-toast";
import { sendAdminMessage } from "@/lib/api-admin";
import type { AdminUserRow } from "@/lib/admin-user-registry";
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
// page. Carries Approve, Reject, Suspend, Unsuspend, Reset verification,
// Send message. Modal state is local so the detail page stays a thin shell.
//
// One-fresh-per-viewport: only Approve is bg-fresh in its rest state.
// Destructive actions (Reject, Suspend) carry the shared BTN_DANGER recipe
// (error text on bone-deep) so they read as consequential before the modal
// opens; restorative/neutral actions stay secondary/ghost.
export function AdminActionAside({ row }: AdminActionAsideProps) {
  const { showToast } = useToast();
  const { approve, reject, suspend, unsuspend, resetVerification } = useAdminUserActions();
  const busy = useRef(false);
  const [submitting, setSubmitting] = useState(false);
  async function run(action: () => Promise<boolean>) {
    if (busy.current) return false;
    busy.current = true; setSubmitting(true);
    try { return await action(); }
    finally { busy.current = false; setSubmitting(false); }
  }

  const [rejectOpen, setRejectOpen] = useState(false);
  const [suspendOpen, setSuspendOpen] = useState(false);
  const [messageOpen, setMessageOpen] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);

  const isPending = row.verificationStatus === "pending";
  const isSuspended = row.suspendedAt !== null;
  const isIncomplete = row.verificationStatus === "incomplete";
  const displayName = row.brandName ?? row.name;

  async function handleApprove() {
    const ok = await run(() => approve(row.userId));
    if (ok) {
      showToast({ kind: "success", message: `Approved · ${displayName}` });
    }
  }

  async function handleReject(reason: string) {
    const ok = await run(() => reject(row.userId, reason));
    if (ok) {
      setRejectOpen(false);
      showToast({ kind: "info", message: `Sent back · ${displayName}` });
    }
  }

  async function handleSuspend(reason: string) {
    const ok = await run(() => suspend(row.userId, reason));
    if (ok) {
      setSuspendOpen(false);
      showToast({ kind: "info", message: `Suspended · ${displayName}` });
    }
  }

  async function handleUnsuspend() {
    const ok = await run(() => unsuspend(row.userId));
    if (ok) {
      showToast({ kind: "success", message: `Unsuspended · ${displayName}` });
    }
  }

  async function handleReset(reason: string) {
    const ok = await run(() => resetVerification(row.userId, reason));
    if (ok) {
      setResetOpen(false);
      showToast({
        kind: "info",
        message: `Verification reset · ${displayName}`,
      });
    }
  }

  async function handleMessage(payload: { subject: string; body: string }) {
    setMessageOpen(false);
    try {
      await sendAdminMessage(row.userId, payload);
      showToast({
        kind: "success",
        message: `Message sent · ${displayName}`,
      });
    } catch (err) {
      console.error("[admin/users] sendMessage failed", err);
      showToast({
        kind: "error",
        message: `Couldn't send message — please try again.`,
      });
    }
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
        {isPending && (
          <>
            <button
              type="button"
              disabled={submitting}
              onClick={handleApprove}
              className={primaryFreshBtn}
            >
              Approve
            </button>
            <button
              type="button"
              disabled={submitting}
              onClick={() => setRejectOpen(true)}
              className={dangerBtn}
            >
              Reject…
            </button>
          </>
        )}

        {!isSuspended ? (
          <button
            type="button"
            disabled={submitting}
            onClick={() => setSuspendOpen(true)}
            className={dangerBtn}
          >
            Suspend…
          </button>
        ) : (
          <button
            type="button"
            disabled={submitting}
            onClick={handleUnsuspend}
            className={secondaryBtn}
          >
            Unsuspend
          </button>
        )}

        {!isIncomplete && (
          <button
            type="button"
            disabled={submitting}
            onClick={() => setResetOpen(true)}
            className={secondaryBtn}
          >
            Reset verification…
          </button>
        )}

        <button
          type="button"
          disabled={submitting}
          onClick={() => setMessageOpen(true)}
          className={tertiaryBtn}
        >
          Send message…
        </button>
      </div>

      {isSuspended && row.suspensionReason && (
        <div className="mt-6 pt-5 border-t border-line">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Suspension reason
          </p>
          <p className="font-sans text-sm text-ink-soft mt-2">
            {row.suspensionReason}
          </p>
        </div>
      )}

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
        open={messageOpen}
        onClose={() => setMessageOpen(false)}
        onSubmit={handleMessage}
        photographerName={displayName}
      />
    </aside>
  );
}

const primaryFreshBtn = `${BTN_PRIMARY} ${BTN_SIZE.sm} w-full`;
const secondaryBtn = `${BTN_SECONDARY} ${BTN_SIZE.sm} w-full`;
const tertiaryBtn = `${BTN_GHOST} ${BTN_SIZE.sm} w-full`;
const dangerBtn = `${BTN_DANGER} ${BTN_SIZE.sm} w-full`;
