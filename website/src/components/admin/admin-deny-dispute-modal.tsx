"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFormSection,
  AdminRadioGroup,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";
import { useDisputeDenyReasons } from "@/hooks/use-admin-disputes";

interface AdminDenyDisputeModalProps {
  open: boolean;
  submitting?: boolean;
  onClose: () => void;
  onSubmit: (reasonCode: string, note: string | null) => void;
  disputeId: string;
  /** Preselected reason code (the dispute's refundBlockedReason); ignored when it is not in the server's list. */
  defaultCode?: string | null;
}

// The one deny path. The reasons come from the server (codes only here); the server
// stores the runner-readable text, so the preview below is exactly what the runner reads.
export function AdminDenyDisputeModal({
  open,
  submitting = false,
  onClose,
  onSubmit,
  disputeId,
  defaultCode = null,
}: AdminDenyDisputeModalProps) {
  const reasons = useDisputeDenyReasons(open);
  const [picked, setPicked] = useState<string | null>(null);
  const [note, setNote] = useState("");

  useEffect(() => {
    if (!open) {
      setNote("");
      setPicked(null);
    }
  }, [open]);

  const list = reasons.data ?? [];
  const code = picked ?? (defaultCode && list.some((r) => r.code === defaultCode) ? defaultCode : null);
  const reason = list.find((r) => r.code === code);
  const trimmed = note.trim();
  const noteMissing = !!reason?.noteRequired && trimmed.length === 0;
  const runnerText = reason ? [reason.runnerText, trimmed].filter(Boolean).join(" ") : "";

  return (
    <AdminFormModal
      open={open}
      submitting={submitting}
      pendingLabel="Denying…"
      onClose={onClose}
      onSubmit={() => reason && onSubmit(reason.code, trimmed || null)}
      title={`Deny ${disputeId}`}
      intro="No refund is issued. The runner sees the reason in their orders ledger and can re-file with new evidence while the refund window is open."
      submitLabel="Deny claim"
      submitDisabled={!reason || noteMissing}
    >
      <AdminFormSection label="Reason">
        {reasons.isPending ? (
          <p role="status" className="font-sans text-sm text-slate-soft">Loading reasons…</p>
        ) : reasons.isError ? (
          <div role="alert" className="space-y-2 font-sans text-sm text-ink-soft">
            <p>Couldn&apos;t load the deny reasons.</p>
            <button
              type="button"
              onClick={() => void reasons.refetch()}
              className="font-mono uppercase tracking-[0.14em] text-[13px] text-ink underline"
            >
              Retry
            </button>
          </div>
        ) : (
          <AdminRadioGroup
            name="dispute-deny-reason"
            options={list.map((r) => ({ value: r.code, label: r.label }))}
            value={code ?? ""}
            onChange={setPicked}
          />
        )}
      </AdminFormSection>
      <AdminTextarea
        id="deny-note"
        label={reason?.noteRequired ? "Note (required, max 280)" : "Note (optional, max 280)"}
        value={note}
        onChange={setNote}
        maxLength={280}
        placeholder="What context should the runner see?"
      />
      {runnerText && (
        <p className="font-sans text-sm text-ink-soft">
          <span className="text-slate-soft">The runner will read: </span>
          {runnerText}
        </p>
      )}
    </AdminFormModal>
  );
}
