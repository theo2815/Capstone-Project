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
  onSubmit: (payload: { subject: string; body: string }) => void;
  photographerName: string;
}

// Send-message modal. On submit, the parent pushes the message into the
// photographer's inbox via notifyAdminMessage(). Audit-log persistence
// arrives with backend Phase F; until then the inbox row is the receipt.
export function AdminMessageModal({
  open,
  onClose,
  onSubmit,
  photographerName,
}: AdminMessageModalProps) {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");

  const subjectValid = subject.trim().length > 0;
  const bodyValid = body.trim().length > 0;
  const canSend = subjectValid && bodyValid;

  function handleSubmit() {
    if (!canSend) return;
    onSubmit({ subject: subject.trim(), body: body.trim() });
    setSubject("");
    setBody("");
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Send message"
      intro={`Reach out to ${photographerName} privately. The message lands directly in their dashboard inbox.`}
      submitLabel="Send"
      submitDisabled={!canSend}
    >
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
