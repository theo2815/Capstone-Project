"use client";

import { useState } from "react";
import { useAdminUserStore } from "@/store/admin-user-store";
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
  const approve = useAdminUserStore((s) => s.approve);
  const reject = useAdminUserStore((s) => s.reject);
  const suspend = useAdminUserStore((s) => s.suspend);
  const unsuspend = useAdminUserStore((s) => s.unsuspend);
  const resetVerification = useAdminUserStore((s) => s.resetVerification);

  const [rejectOpen, setRejectOpen] = useState(false);
  const [suspendOpen, setSuspendOpen] = useState(false);
  const [messageOpen, setMessageOpen] = useState(false);
  const [resetOpen, setResetOpen] = useState(false);

  const isPending = row.verificationStatus === "pending";
  const isSuspended = row.suspendedAt !== null;
  const isIncomplete = row.verificationStatus === "incomplete";
  const displayName = row.brandName ?? row.name;

  // Each handler awaits the store and only fires the success toast on BE
  // success — the photographer notification is fired inside the store
  // action, so we never get a "you're approved!" inbox row on a failed
  // approve. The store also surfaces its own error toast on failure, so
  // the caller need not handle the false branch.
  async function handleApprove() {
    const ok = await approve(row.userId);
    if (ok) {
      showToast({ kind: "success", message: `Approved · ${displayName}` });
    }
  }

  async function handleReject(reason: string) {
    const ok = await reject(row.userId, reason);
    setRejectOpen(false);
    if (ok) {
      showToast({ kind: "info", message: `Sent back · ${displayName}` });
    }
  }

  async function handleSuspend(reason: string) {
    const ok = await suspend(row.userId, reason);
    setSuspendOpen(false);
    if (ok) {
      showToast({ kind: "info", message: `Suspended · ${displayName}` });
    }
  }

  async function handleUnsuspend() {
    const ok = await unsuspend(row.userId);
    if (ok) {
      showToast({ kind: "success", message: `Unsuspended · ${displayName}` });
    }
  }

  async function handleReset(reason: string) {
    setResetOpen(false);
    const ok = await resetVerification(row.userId, reason);
    if (ok) {
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
              onClick={handleApprove}
              className={primaryFreshBtn}
            >
              Approve
            </button>
            <button
              type="button"
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
            onClick={() => setSuspendOpen(true)}
            className={dangerBtn}
          >
            Suspend…
          </button>
        ) : (
          <button
            type="button"
            onClick={handleUnsuspend}
            className={secondaryBtn}
          >
            Unsuspend
          </button>
        )}

        {!isIncomplete && (
          <button
            type="button"
            onClick={() => setResetOpen(true)}
            className={secondaryBtn}
          >
            Reset verification…
          </button>
        )}

        <button
          type="button"
          onClick={() => setMessageOpen(true)}
          className={tertiaryBtn}
        >
          Send message…
        </button>
      </div>

      {isSuspended && row.suspensionReason && (
        <div className="mt-6 pt-5 border-t border-line">
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
            Suspension reason
          </p>
          <p className="font-sans text-sm text-ink-soft mt-2">
            {row.suspensionReason}
          </p>
        </div>
      )}

      <AdminRejectModal
        open={rejectOpen}
        onClose={() => setRejectOpen(false)}
        onSubmit={handleReject}
        photographerName={displayName}
      />
      <AdminResetVerificationModal
        open={resetOpen}
        onClose={() => setResetOpen(false)}
        onSubmit={handleReset}
        photographerName={displayName}
      />
      <AdminSuspendModal
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
