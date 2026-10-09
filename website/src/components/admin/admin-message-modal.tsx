"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminTextInput,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminMessageModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (payload: { subject: string; body: string }) => Promise<boolean>;
  submitting: boolean;
  recoveryNotice?: string | null;
  photographerName: string;
}

// Drafts stay in memory until the domain hook confirms the keyed send.
export function AdminMessageModal({
  open,
  onClose,
  onSubmit,
  photographerName,
  submitting,
  recoveryNotice,
}: AdminMessageModalProps) {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");

  const subjectValid = subject.trim().length > 0;
  const bodyValid = body.trim().length > 0;
  const canSend = subjectValid && bodyValid;

  async function handleSubmit() {
    if (!canSend || submitting) return;
    if (await onSubmit({ subject: subject.trim(), body: body.trim() })) {
      setSubject("");
      setBody("");
    }
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Send message"
      intro={`Reach out to ${photographerName} privately. The message lands directly in their dashboard inbox.`}
      submitLabel="Send"
      submitting={submitting}
      pendingLabel="Sending…"
      submitDisabled={!canSend}
    >
      {recoveryNotice && <p role="status" className="font-sans text-sm text-slate">{recoveryNotice}</p>}
      <AdminTextInput
        id="msg-subject"
        label="Subject"
        value={subject}
        onChange={setSubject}
        maxLength={80}
        showCounter
        placeholder="Quick question about your latest upload"
      />
      <AdminTextarea
        id="msg-body"
        label="Message"
        value={body}
        onChange={setBody}
        maxLength={600}
        rows={5}
        placeholder="Hey, just wanted to flag…"
      />
    </AdminFormModal>
  );
}
