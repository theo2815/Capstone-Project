"use client";

import { useState } from "react";
import { useAdminUserStore } from "@/store/admin-user-store";
import { useToast } from "@/hooks/use-toast";
import {
  notifyAccountSuspended,
  notifyAccountUnsuspended,
  notifyAdminMessage,
  notifyVerificationApproved,
  notifyVerificationRejected,
} from "@/lib/admin-photographer-notifications";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { AdminRejectModal } from "./admin-reject-modal";
import { AdminSuspendModal } from "./admin-suspend-modal";
import { AdminEditPhotographerModal } from "./admin-edit-photographer-modal";
import { AdminMessageModal } from "./admin-message-modal";
import { cn } from "@/lib/utils";

interface AdminActionAsideProps {
  row: AdminUserRow;
}

// Sticky-on-lg / inline-on-md action cluster for the photographer detail
// page. Carries all 7 actions (Approve, Reject, Suspend, Unsuspend, Force-
// edit, Reset verification, Send message). Modal state is local so the
// detail page stays a thin shell.
//
// One-fresh-per-viewport: only Approve is bg-fresh in its rest state. All
// other primary buttons rest at border-line text-ink and roll to bg-ink on
// hover. Reject is a tertiary button (no border).
export function AdminActionAside({ row }: AdminActionAsideProps) {
  const { showToast } = useToast();
  const approve = useAdminUserStore((s) => s.approve);
  const reject = useAdminUserStore((s) => s.reject);
  const suspend = useAdminUserStore((s) => s.suspend);
  const unsuspend = useAdminUserStore((s) => s.unsuspend);
  const forceEdit = useAdminUserStore((s) => s.forceEdit);
  const resetVerification = useAdminUserStore((s) => s.resetVerification);

  const [rejectOpen, setRejectOpen] = useState(false);
  const [suspendOpen, setSuspendOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [messageOpen, setMessageOpen] = useState(false);
  const [resetConfirm, setResetConfirm] = useState(false);

  const isPending = row.verificationStatus === "pending";
  const isSuspended = row.suspendedAt !== null;
  const isIncomplete = row.verificationStatus === "incomplete";
  const displayName = row.brandName ?? row.name;

  function handleApprove() {
    approve(row.userId);
    notifyVerificationApproved({
      photographerId: row.userId,
      brandName: displayName,
    });
    showToast({ kind: "success", message: `Approved · ${displayName}` });
  }

  function handleReject(reason: string) {
    reject(row.userId, reason);
    notifyVerificationRejected({
      photographerId: row.userId,
      brandName: displayName,
      reason,
    });
    setRejectOpen(false);
    showToast({ kind: "info", message: `Sent back · ${displayName}` });
  }

  function handleSuspend(reason: string) {
    suspend(row.userId, reason);
    notifyAccountSuspended({
      photographerId: row.userId,
      brandName: displayName,
      reason,
    });
    setSuspendOpen(false);
    showToast({ kind: "info", message: `Suspended · ${displayName}` });
  }

  function handleUnsuspend() {
    unsuspend(row.userId);
    notifyAccountUnsuspended({
      photographerId: row.userId,
      brandName: displayName,
    });
    showToast({ kind: "success", message: `Unsuspended · ${displayName}` });
  }

  function handleForceEdit(patch: {
    handle: string | null;
    brandName: string | null;
  }) {
    forceEdit(row.userId, patch);
    setEditOpen(false);
    showToast({ kind: "info", message: `Profile updated · ${displayName}` });
  }

  function handleReset() {
    resetVerification(row.userId);
    setResetConfirm(false);
    showToast({
      kind: "info",
      message: `Verification reset · ${displayName}`,
    });
  }

  function handleMessage(payload: { subject: string; body: string }) {
    notifyAdminMessage({
      photographerId: row.userId,
      subject: payload.subject,
      body: payload.body,
    });
    setMessageOpen(false);
    showToast({
      kind: "success",
      message: `Message sent · ${displayName}`,
    });
  }

  return (
    <aside
      aria-label="Admin actions"
      className="lg:sticky lg:top-24 mt-12 lg:mt-0 rounded-2xl border border-line bg-bone p-5 md:p-6"
    >
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        Admin actions
      </p>
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
              className={tertiaryBtn}
            >
              Reject…
            </button>
          </>
        )}

        {!isSuspended ? (
          <button
            type="button"
            onClick={() => setSuspendOpen(true)}
            className={cn(secondaryBtn, "border-ink/20")}
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

        <button
          type="button"
          onClick={() => setEditOpen(true)}
          className={secondaryBtn}
        >
          Force-edit profile…
        </button>

        {!isIncomplete && (
          <button
            type="button"
            onClick={() => setResetConfirm((v) => !v)}
            aria-expanded={resetConfirm}
            className={secondaryBtn}
          >
            {resetConfirm ? "Cancel reset" : "Reset verification"}
          </button>
        )}
        {resetConfirm && !isIncomplete && (
          <div className="rounded-xl border border-line bg-bone-deep p-4 space-y-3">
            <p className="font-sans text-sm text-ink-soft">
              Drops {displayName} back to <strong className="text-ink">incomplete</strong>.
              They&apos;ll need to resubmit settings for review.
            </p>
            <button
              type="button"
              onClick={handleReset}
              className={cn(secondaryBtn, "w-full")}
            >
              Confirm reset
            </button>
          </div>
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
      <AdminSuspendModal
        open={suspendOpen}
        onClose={() => setSuspendOpen(false)}
        onSubmit={handleSuspend}
        photographerName={displayName}
      />
      <AdminEditPhotographerModal
        open={editOpen}
        onClose={() => setEditOpen(false)}
        onSubmit={handleForceEdit}
        initialHandle={row.handle}
        initialBrandName={row.brandName}
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

const primaryFreshBtn =
  "w-full font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] bg-fresh text-bone hover:bg-fresh-deep transition-colors rounded-full px-5 py-2.5";

const secondaryBtn =
  "w-full font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2.5";

const tertiaryBtn =
  "w-full font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-5 py-2.5";
